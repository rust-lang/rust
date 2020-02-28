#[warn(clippy::suboptimal_flops)]

fn fake_abs1(num: f64) -> f64 {
    if num >= 0.0 {
        num
    } else {
        -num
    }
}

fn fake_abs2(num: f64) -> f64 {
    if 0.0 < num {
        num
    } else {
        -num
    }
}

fn fake_nabs1(num: f64) -> f64 {
    if num < 0.0 {
        num
    } else {
        -num
    }
}

fn fake_nabs2(num: f64) -> f64 {
    if 0.0 >= num {
        num
    } else {
        -num
    }
}

fn not_fake_abs1(num: f64) -> f64 {
    if num > 0.0 {
        num
    } else {
        -num - 1f64
    }
}

fn not_fake_abs2(num: f64) -> f64 {
    if num > 0.0 {
        num + 1.0
    } else {
        -(num + 1.0)
    }
}

fn not_fake_abs3(num1: f64, num2: f64) -> f64 {
    if num1 > 0.0 {
        num2
    } else {
        -num2
    }
}

fn main() {}
