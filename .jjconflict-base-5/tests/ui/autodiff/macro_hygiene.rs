//@ compile-flags: -Zautodiff=Enable -C opt-level=3  -Clto=fat
//@ no-prefer-dynamic
//@ needs-enzyme
//@ check-pass

// In the past, we just checked for correct macro hygiene information.

#![feature(autodiff)]

macro_rules! demo {
    () => {
        #[std::autodiff::autodiff_reverse(fd, Active, Active)]
        fn f(x: f64) -> f64 {
            x * x
        }
    };
}
demo!();

fn main() {
    dbg!(f(2.0f64));
}
