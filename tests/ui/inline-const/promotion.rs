#![feature(inline_const)]
#![allow(arithmetic_overflow, unconditional_panic)]
// check-pass

// The only way to have promoteds that fail is in `const fn` called from `const`/`static`.
const fn div_by_zero() -> i32 {
    1 / 0
}

const fn mk_false() -> bool {
    false
}

fn main() {
    let v = const {
        if mk_false() {
            let _x: &'static i32 = &div_by_zero();
        }
        42
    };
}
