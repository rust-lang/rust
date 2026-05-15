//@ revisions: bpass1 bpass2
//@ compile-flags: -Z query-dep-graph -Copt-level=0
// FIXME(#62277): could be check-pass?

#![allow(warnings)]
#![feature(rustc_attrs)]
#![crate_type = "rlib"]

// Here the only thing which changes is the string constant in `x`.
// Therefore, the compiler deduces (correctly) that typeck_root is not
// needed even for callers of `x`.

pub mod x {
    #[cfg(bpass1)]
    pub fn x() {
        println!("{}", "1");
    }

    #[cfg(bpass2)]
    #[rustc_clean(except = "owner,optimized_mir", cfg = "bpass2")]
    pub fn x() {
        println!("{}", "2");
    }
}

pub mod y {
    use x;

    #[rustc_clean(cfg = "bpass2")]
    pub fn y() {
        x::x();
    }
}

pub mod z {
    use y;

    #[rustc_clean(cfg = "bpass2")]
    pub fn z() {
        y::y();
    }
}
