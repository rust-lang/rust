// Check that ref mut patterns within a const pattern don't get considered
// unsafe because they're within a pattern for a layout constrained stuct.
//@ check-pass

#![feature(rustc_attrs)]
#![feature(inline_const_pat)]

#[rustc_layout_scalar_valid_range_start(3)]
struct Gt2(i32);

fn main() {
    match unsafe { Gt2(5) } {
        Gt2(
            const {
                || match () {
                    ref mut y => (),
                };
                4
            },
        ) => (),
        _ => (),
    }
}
