// Compiler:
//
// Run-time:
//   status: 0

#![feature(const_black_box)]

fn main() {
    use std::hint::black_box;

    macro_rules! check {
        ($ty:ty, $expr:expr) => {{
            const EXPECTED: $ty = $expr;
            assert_eq!($expr, EXPECTED);
        }};
    }

    check!(i32, (black_box(0.0f32) as i32));

    check!(u64, (black_box(f32::NAN) as u64));
    check!(u128, (black_box(f32::NAN) as u128));

    check!(i64, (black_box(f64::NAN) as i64));
    check!(u64, (black_box(f64::NAN) as u64));

    check!(i16, (black_box(f32::MIN) as i16));
    check!(i16, (black_box(f32::MAX) as i16));
}
