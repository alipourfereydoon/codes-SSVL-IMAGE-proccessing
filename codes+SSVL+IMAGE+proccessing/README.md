# پردازش تصویر با مقایسه‌ی CMOS، نقطه‌کوانتومی و SSVL

## فارسی

این پروژه یک شبیه‌ساز آموزشی و بازتولیدپذیر برای مقایسه‌ی آشکارسازی لبه در سه الگو است: CMOS کلاسیک، خوانش شارژِ نقطه‌کوانتومی (QD) و QD تقویت‌شده با لایه‌ی راستی‌آزمایی امضای اسپین (SSVL). مدل، پدیده‌ها را به‌صورت آماری/پدیدارشناختی پیاده‌سازی می‌کند و **حل عددی هامیلتونی یا شبیه‌ساز کامل دینامیک کوانتومی نیست**.

### نصب و اجرا

Python 3.9 یا جدیدتر توصیه می‌شود:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_simulation.py
```

اجرای تصویر دلخواه (ورودی با Pillow خوانده و خاکستری می‌شود):

```bash
python run_simulation.py --image ./my_image.png --output ./my_results --seed 21
```

گزینه‌ها:

- `--image PATH`: تصویر دلخواه؛ در صورت حذف، فانتوم هندسی ساخته می‌شود.
- `--size N`: اندازه‌ی فانتوم مربعی (پیش‌فرض 256؛ حداقل 32).
- `--output DIR`: پوشه‌ی خروجی (پیش‌فرض `results`).
- `--seed N`: بذر تصادفی برای تکرارپذیری.
- `--jitter X`: شدت نوسان مسیر شارژ، در بازه‌ی 0 تا 1 (پیش‌فرض 0.10).
- `--parameter-variance X`: ناهمگنی پارامترهای محلی (پیش‌فرض 0.04).
- `--bit-flip X`: احتمال پایه‌ی خطای بیت (پیش‌فرض 0.015).
- `--delta-t-over-t X`: انحراف زمانی نرمال‌شده‌ی نامی برای اسپین (پیش‌فرض 0.15).
- `--spin-noise X`: انحراف معیار نویز اندازه‌گیری فیدلیتی (پیش‌فرض 0.015).
- `--edge-threshold X`: آستانه‌ی مطلق گرادیان خاکستری؛ بدون آن آستانه از داده تعیین می‌شود.

خروجی‌ها در پوشه‌ی انتخابی: `metrics.json`، `metrics.csv`، `comparison.png` و `ssvl_summary.json`.

### اجزای مدل و فرمول‌ها

- **ImageDataset**: بارگذاری تصویر خاکستری یا تولید فانتومِ دایره/مستطیل/مثلث/خط و مرجع لبه.
- **EdgeDetectionKernel**: اختلاف‌های جهت‌دار افقی و عمودی را آستانه‌گذاری می‌کند؛ منطق XOR برای اختلاف جهت‌ها و OR برای نگه‌داشتن گوشه‌ها به‌کار می‌رود.
- **QuantumDotSimulator**: نویز مکانی پارامترها و احتمال خطای مثبت/منفی را به لبه‌ها می‌افزاید. احتمال‌ها به‌طور تقریبی تابع jitter و bit-flip هستند؛ این‌ها نرخ‌های پدیدارشناختی‌اند، نه نرخ‌های حاصل از حل هامیلتونی.
- **SSVLVerifier**: حالت کمکی اسپین را با فیدلیتی هم‌پوشانی رامزی مدل می‌کند:

  `F_S = clip(1 - 0.33*(deltaT/T)^2 + epsilon, 0, 1)`, با `epsilon ~ Normal(0, spin_noise)`.

  رأی اسپین با آستانه‌ی تطبیقی و پشتیبانی همسایگی با خروجی شارژ تلفیق می‌شود تا تصمیم اجماعی/کاهش خطا بسازد.
- **MetricsEvaluator**: PSNR بر نقشه‌ی دودویی لبه با دامنه‌ی داده‌ی 1، SSIM، نرخ مثبت کاذب `FP/(FP+TN)`، نرخ منفی کاذب `FN/(FN+TP)` و AUC رتبه‌ای را گزارش می‌کند. AUC با امتیاز پیوسته‌ی هر روش محاسبه می‌شود؛ اگر حقیقت فقط یک کلاس داشته باشد، AUC برابر NaN است.

### نگاشت به چارچوب پایان‌نامه

پیاده‌سازی، ایده‌ی «منطق شارژ + حالت کمکی اسپین برای راستی‌آزمایی مسیر» را به یک آزمایش تصویری قابل‌اجرا نگاشت می‌کند: لبه‌ها نقش رویدادهای منطقی، خطای تونل‌زنی/نویز شارژ نقش خطاهای اجرا، و فیدلیتی اسپین/اجماع نقش نشانگر تشخیص و کاهش خطا را دارند. رابطه‌ی درجه‌دوم فیدلیتی با `deltaT/T` از رابطه‌ی تحلیلی SSVL الهام گرفته شده است. در این پروژه از جمله موارد زیر بازسازی نشده‌اند: فضای هیلبرت، مدل Hubbard توسعه‌یافته، تبدیل Schrieffer–Wolff، تکامل سامانه‌ی باز، کوپلینگ فیزیکی بار-اسپین یا ارقام تجربی/شبیه‌سازی مقاله. نتایج نمونه را نباید به‌عنوان تأیید آن پیش‌بینی‌های فیزیکی تفسیر کرد.

## English

This self-contained teaching simulator compares edge detection under three paradigms: classical CMOS, charge-only quantum-dot (QD) readout, and SSVL-enhanced QD readout. It is a **phenomenological statistical model**, not a microscopic Hamiltonian solver or a full quantum-dynamics simulator.

### Install and run

Python 3.9+ is recommended:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_simulation.py
```

Use a custom image (Pillow loads and converts it to grayscale):

```bash
python run_simulation.py --image ./my_image.png --output ./my_results --seed 21
```

CLI arguments: `--image PATH` (omit for synthetic phantom), `--size N` (phantom side, default 256; min 32), `--output DIR` (default `results`), `--seed N`, `--jitter X` (0–1, default .10), `--parameter-variance X` (default .04), `--bit-flip X` (default .015), `--delta-t-over-t X` (default .15), `--spin-noise X` (default .015), and optional absolute `--edge-threshold X` (otherwise data-adaptive). Outputs are `metrics.json`, `metrics.csv`, `comparison.png`, and `ssvl_summary.json`.

### Components and formulas

- **ImageDataset** loads a grayscale image or generates a circle/rectangle/triangle/line phantom and edge reference.
- **EdgeDetectionKernel** thresholds horizontal and vertical directional differences. XOR expresses directional parity; OR retains corner edges.
- **QuantumDotSimulator** adds spatial parameter variation and stochastic false/missed edges. Its probabilities are phenomenological functions of jitter and bit-flip settings, not rates derived from a physical Hamiltonian.
- **SSVLVerifier** represents auxiliary-spin preparation/readout with Ramsey-overlap fidelity:

  `F_S = clip(1 - 0.33*(deltaT/T)^2 + epsilon, 0, 1)`, where `epsilon ~ Normal(0, spin_noise)`.

  A data-adaptive spin threshold and spatial consensus are combined with charge confidence for in-situ error mitigation.
- **MetricsEvaluator** reports binary-edge PSNR (data range 1), SSIM, false-positive rate `FP/(FP+TN)`, false-negative rate `FN/(FN+TP)`, and rank-based AUC from continuous scores. AUC is NaN if truth contains only one class.

### Mapping to the thesis framework

The image experiment maps “charge logic plus an auxiliary spin state that verifies the trajectory” to edge events: edges represent logical events, stochastic charge errors represent faulty executions, and spin fidelity/consensus represents error indication and mitigation. The quadratic fidelity law follows the stated SSVL analytical form. This project does **not** reproduce the Hilbert-space model, extended Hubbard Hamiltonian, Schrieffer–Wolff transformation, open-system dynamics, physical charge-spin coupling, or reported numerical/experimental results. Its output must not be interpreted as validation of those physical predictions.
