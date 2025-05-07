#![warn(clippy::manual_abs_diff)]

use std::time::Duration;

fn main() {
    let a: usize = 5;
    let b: usize = 3;
    let c: usize = 8;
    let d: usize = 11;

    let _ = if a > b { a - b } else { b - a };
    //~^ manual_abs_diff
    let _ = if a < b { b - a } else { a - b };
    //~^ manual_abs_diff

    let _ = if 5 > b { 5 - b } else { b - 5 };
    //~^ manual_abs_diff
    let _ = if b > 5 { b - 5 } else { 5 - b };
    //~^ manual_abs_diff

    let _ = if a >= b { a - b } else { b - a };
    //~^ manual_abs_diff
    let _ = if a <= b { b - a } else { a - b };
    //~^ manual_abs_diff

    #[allow(arithmetic_overflow)]
    {
        let _ = if a > b { b - a } else { a - b };
        let _ = if a < b { a - b } else { b - a };
    }

    let _ = if (a + b) > (c + d) {
        //~^ manual_abs_diff
        (a + b) - (c + d)
    } else {
        (c + d) - (a + b)
    };
    let _ = if (a + b) < (c + d) {
        //~^ manual_abs_diff
        (c + d) - (a + b)
    } else {
        (a + b) - (c + d)
    };

    const A: usize = 5;
    const B: usize = 3;
    // check const context
    const _: usize = if A > B { A - B } else { B - A };
    //~^ manual_abs_diff

    let a = Duration::from_secs(3);
    let b = Duration::from_secs(5);
    let _ = if a > b { a - b } else { b - a };
    //~^ manual_abs_diff

    let a: i32 = 3;
    let b: i32 = -5;
    let _ = if a > b { a - b } else { b - a };
    let _ = if a > b { (a - b) as u32 } else { (b - a) as u32 };
    //~^ manual_abs_diff
}

// FIXME: bunch of patterns that should be linted
fn fixme() {
    let a: usize = 5;
    let b: usize = 3;
    let c: usize = 8;
    let d: usize = 11;

    {
        let out;
        if a > b {
            out = a - b;
        } else {
            out = b - a;
        }
    }

    {
        let mut out = 0;
        if a > b {
            out = a - b;
        } else if a < b {
            out = b - a;
        }
    }

    #[allow(clippy::implicit_saturating_sub)]
    let _ = if a > b {
        a - b
    } else if a < b {
        b - a
    } else {
        0
    };

    let a: i32 = 3;
    let b: i32 = 5;
    let _: u32 = if a > b { a - b } else { b - a } as u32;
}

fn non_primitive_ty() {
    #[derive(Eq, PartialEq, PartialOrd)]
    struct S(i32);

    impl std::ops::Sub for S {
        type Output = S;

        fn sub(self, rhs: Self) -> Self::Output {
            Self(self.0 - rhs.0)
        }
    }

    let (a, b) = (S(10), S(20));
    let _ = if a < b { b - a } else { a - b };
}
