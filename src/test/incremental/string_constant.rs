// revisions: cfail1 cfail2 cfail3 cfail4
// compile-flags: -Z query-dep-graph
// [cfail3]compile-flags: -Zincremental-relative-spans
// [cfail4]compile-flags: -Zincremental-relative-spans
// build-pass (FIXME(62277): could be check-pass?)

#![allow(warnings)]
#![feature(rustc_attrs)]
#![crate_type = "rlib"]

// Here the only thing which changes is the string constant in `x`.
// Therefore, the compiler deduces (correctly) that typeck is not
// needed even for callers of `x`.

pub mod x {
    #[cfg(any(cfail1, cfail3))]
    pub fn x() {
        println!("{}", "1");
    }

    #[cfg(any(cfail2, cfail4))]
    #[rustc_clean(except = "hir_owner,hir_owner_nodes,optimized_mir,promoted_mir", cfg = "cfail2")]
    #[rustc_clean(except = "hir_owner_nodes,promoted_mir", cfg = "cfail4")]
    pub fn x() {
        println!("{}", "2");
    }
}

pub mod y {
    use x;

    #[rustc_clean(cfg = "cfail2")]
    #[rustc_clean(cfg = "cfail4")]
    pub fn y() {
        x::x();
    }
}

pub mod z {
    use y;

    #[rustc_clean(cfg = "cfail2")]
    #[rustc_clean(cfg = "cfail4")]
    pub fn z() {
        y::y();
    }
}
