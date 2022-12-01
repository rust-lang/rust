#![warn(clippy::manual_clamp)]
#![allow(
    unused,
    dead_code,
    clippy::unnecessary_operation,
    clippy::no_effect,
    clippy::if_same_then_else
)]

use std::cmp::{max as cmp_max, min as cmp_min};

const CONST_MAX: i32 = 10;
const CONST_MIN: i32 = 4;

const CONST_F64_MAX: f64 = 10.0;
const CONST_F64_MIN: f64 = 4.0;

fn main() {
    let (input, min, max) = (0, -2, 3);
    // Lint
    let x0 = if max < input {
        max
    } else if min > input {
        min
    } else {
        input
    };

    let x1 = if input > max {
        max
    } else if input < min {
        min
    } else {
        input
    };

    let x2 = if input < min {
        min
    } else if input > max {
        max
    } else {
        input
    };

    let x3 = if min > input {
        min
    } else if max < input {
        max
    } else {
        input
    };

    let x4 = input.max(min).min(max);

    let x5 = input.min(max).max(min);

    let x6 = match input {
        x if x > max => max,
        x if x < min => min,
        x => x,
    };

    let x7 = match input {
        x if x < min => min,
        x if x > max => max,
        x => x,
    };

    let x8 = match input {
        x if max < x => max,
        x if min > x => min,
        x => x,
    };

    let mut x9 = input;
    if x9 < min {
        x9 = min;
    }
    if x9 > max {
        x9 = max;
    }

    let x10 = match input {
        x if min > x => min,
        x if max < x => max,
        x => x,
    };

    let mut x11 = input;
    let _ = 1;
    if x11 > max {
        x11 = max;
    }
    if x11 < min {
        x11 = min;
    }

    let mut x12 = input;
    if min > x12 {
        x12 = min;
    }
    if max < x12 {
        x12 = max;
    }

    let mut x13 = input;
    if max < x13 {
        x13 = max;
    }
    if min > x13 {
        x13 = min;
    }

    let x14 = if input > CONST_MAX {
        CONST_MAX
    } else if input < CONST_MIN {
        CONST_MIN
    } else {
        input
    };
    {
        let (input, min, max) = (0.0f64, -2.0, 3.0);
        let x15 = if input > max {
            max
        } else if input < min {
            min
        } else {
            input
        };
    }
    {
        let input: i32 = cmp_min_max(1);
        // These can only be detected if exactly one of the arguments to the inner function is const.
        let x16 = cmp_max(cmp_min(input, CONST_MAX), CONST_MIN);
        let x17 = cmp_min(cmp_max(input, CONST_MIN), CONST_MAX);
        let x18 = cmp_max(CONST_MIN, cmp_min(input, CONST_MAX));
        let x19 = cmp_min(CONST_MAX, cmp_max(input, CONST_MIN));
        let x20 = cmp_max(cmp_min(CONST_MAX, input), CONST_MIN);
        let x21 = cmp_min(cmp_max(CONST_MIN, input), CONST_MAX);
        let x22 = cmp_max(CONST_MIN, cmp_min(CONST_MAX, input));
        let x23 = cmp_min(CONST_MAX, cmp_max(CONST_MIN, input));
        let input: f64 = cmp_min_max(1) as f64;
        let x24 = f64::max(f64::min(input, CONST_F64_MAX), CONST_F64_MIN);
        let x25 = f64::min(f64::max(input, CONST_F64_MIN), CONST_F64_MAX);
        let x26 = f64::max(CONST_F64_MIN, f64::min(input, CONST_F64_MAX));
        let x27 = f64::min(CONST_F64_MAX, f64::max(input, CONST_F64_MIN));
        let x28 = f64::max(f64::min(CONST_F64_MAX, input), CONST_F64_MIN);
        let x29 = f64::min(f64::max(CONST_F64_MIN, input), CONST_F64_MAX);
        let x30 = f64::max(CONST_F64_MIN, f64::min(CONST_F64_MAX, input));
        let x31 = f64::min(CONST_F64_MAX, f64::max(CONST_F64_MIN, input));
    }
    let mut x32 = input;
    if x32 < min {
        x32 = min;
    } else if x32 > max {
        x32 = max;
    }

    // It's important this be the last set of statements
    let mut x33 = input;
    if max < x33 {
        x33 = max;
    }
    if min > x33 {
        x33 = min;
    }
}

// This code intentionally nonsense.
fn no_lint() {
    let (input, min, max) = (0, -2, 3);
    let x0 = if max < input {
        max
    } else if min > input {
        max
    } else {
        min
    };

    let x1 = if input > max {
        max
    } else if input > min {
        min
    } else {
        max
    };

    let x2 = if max < min {
        min
    } else if input > max {
        input
    } else {
        input
    };

    let x3 = if min > input {
        input
    } else if max < input {
        max
    } else {
        max
    };

    let x6 = match input {
        x if x < max => x,
        x if x < min => x,
        x => x,
    };

    let x7 = match input {
        x if x < min => max,
        x if x > max => min,
        x => x,
    };

    let x8 = match input {
        x if max > x => max,
        x if min > x => min,
        x => x,
    };

    let mut x9 = input;
    if x9 > min {
        x9 = min;
    }
    if x9 > max {
        x9 = max;
    }

    let x10 = match input {
        x if min > x => min,
        x if max < x => max,
        x => min,
    };

    let mut x11 = input;
    if x11 > max {
        x11 = min;
    }
    if x11 < min {
        x11 = max;
    }

    let mut x12 = input;
    if min > x12 {
        x12 = max * 3;
    }
    if max < x12 {
        x12 = min;
    }

    let mut x13 = input;
    if max < x13 {
        let x13 = max;
    }
    if min > x13 {
        x13 = min;
    }
    let mut x14 = input;
    if x14 < min {
        x14 = 3;
    } else if x14 > max {
        x14 = max;
    }
    {
        let input: i32 = cmp_min_max(1);
        // These can only be detected if exactly one of the arguments to the inner function is const.
        let x16 = cmp_max(cmp_max(input, CONST_MAX), CONST_MIN);
        let x17 = cmp_min(cmp_min(input, CONST_MIN), CONST_MAX);
        let x18 = cmp_max(CONST_MIN, cmp_max(input, CONST_MAX));
        let x19 = cmp_min(CONST_MAX, cmp_min(input, CONST_MIN));
        let x20 = cmp_max(cmp_max(CONST_MAX, input), CONST_MIN);
        let x21 = cmp_min(cmp_min(CONST_MIN, input), CONST_MAX);
        let x22 = cmp_max(CONST_MIN, cmp_max(CONST_MAX, input));
        let x23 = cmp_min(CONST_MAX, cmp_min(CONST_MIN, input));
        let input: f64 = cmp_min_max(1) as f64;
        let x24 = f64::max(f64::max(input, CONST_F64_MAX), CONST_F64_MIN);
        let x25 = f64::min(f64::min(input, CONST_F64_MIN), CONST_F64_MAX);
        let x26 = f64::max(CONST_F64_MIN, f64::max(input, CONST_F64_MAX));
        let x27 = f64::min(CONST_F64_MAX, f64::min(input, CONST_F64_MIN));
        let x28 = f64::max(f64::max(CONST_F64_MAX, input), CONST_F64_MIN);
        let x29 = f64::min(f64::min(CONST_F64_MIN, input), CONST_F64_MAX);
        let x30 = f64::max(CONST_F64_MIN, f64::max(CONST_F64_MAX, input));
        let x31 = f64::min(CONST_F64_MAX, f64::min(CONST_F64_MIN, input));
        let x32 = f64::min(CONST_F64_MAX, f64::min(CONST_F64_MIN, CONST_F64_MAX));
    }
}

fn dont_tell_me_what_to_do() {
    let (input, min, max) = (0, -2, 3);
    let mut x_never = input;
    #[allow(clippy::manual_clamp)]
    if x_never < min {
        x_never = min;
    }
    if x_never > max {
        x_never = max;
    }
}

/// Just to ensure this isn't const evaled
fn cmp_min_max(input: i32) -> i32 {
    input * 3
}

#[clippy::msrv = "1.49"]
fn msrv_1_49() {
    let (input, min, max) = (0, -1, 2);
    let _ = if input < min {
        min
    } else if input > max {
        max
    } else {
        input
    };
}

#[clippy::msrv = "1.50"]
fn msrv_1_50() {
    let (input, min, max) = (0, -1, 2);
    let _ = if input < min {
        min
    } else if input > max {
        max
    } else {
        input
    };
}
