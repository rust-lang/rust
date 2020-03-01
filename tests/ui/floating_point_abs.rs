#![warn(clippy::suboptimal_flops)]

struct A {
    a: f64,
    b: f64,
}

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

fn fake_abs3(a: A) -> f64 {
    if a.a > 0.0 {
        a.a
    } else {
        -a.a
    }
}

fn fake_abs4(num: f64) -> f64 {
    if 0.0 >= num {
        -num
    } else {
        num
    }
}

fn fake_abs5(a: A) -> f64 {
    if a.a < 0.0 {
        -a.a
    } else {
        a.a
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

fn fake_nabs3(a: A) -> A {
    A {
        a: if a.a >= 0.0 { -a.a } else { a.a },
        b: a.b,
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

fn not_fake_abs4(a: A) -> f64 {
    if a.a > 0.0 {
        a.b
    } else {
        -a.b
    }
}

fn not_fake_abs5(a: A) -> f64 {
    if a.a > 0.0 {
        a.a
    } else {
        -a.b
    }
}

fn main() {}
