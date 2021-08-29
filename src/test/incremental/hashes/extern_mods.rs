// This test case tests the incremental compilation hash (ICH) implementation
// for `extern` modules.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

// build-pass (FIXME(62277): could be check-pass?)
// revisions: cfail1 cfail2 cfail3 cfail4 cfail5 cfail6
// compile-flags: -Z query-dep-graph
// [cfail1]compile-flags: -Zincremental-ignore-spans
// [cfail2]compile-flags: -Zincremental-ignore-spans
// [cfail3]compile-flags: -Zincremental-ignore-spans
// [cfail4]compile-flags: -Zincremental-relative-spans
// [cfail5]compile-flags: -Zincremental-relative-spans
// [cfail6]compile-flags: -Zincremental-relative-spans

#![allow(warnings)]
#![feature(rustc_attrs)]
#![feature(unboxed_closures)]
#![crate_type = "rlib"]

// Change function name --------------------------------------------------------
#[cfg(any(cfail1,cfail4))]
extern "C" {
    pub fn change_function_name1(c: i64) -> i32;
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg = "cfail2", except = "hir_owner")]
#[rustc_clean(cfg = "cfail3")]
#[rustc_clean(cfg = "cfail5", except = "hir_owner")]
#[rustc_clean(cfg = "cfail6")]
extern "C" {
    pub fn change_function_name2(c: i64) -> i32;
}

// Change parameter name -------------------------------------------------------
#[cfg(any(cfail1,cfail4))]
extern "C" {
    pub fn change_parameter_name(c: i64) -> i32;
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg = "cfail2")]
#[rustc_clean(cfg = "cfail3")]
#[rustc_clean(cfg = "cfail5")]
#[rustc_clean(cfg = "cfail6")]
extern "C" {
    pub fn change_parameter_name(d: i64) -> i32;
}

// Change parameter type -------------------------------------------------------
#[cfg(any(cfail1,cfail4))]
extern "C" {
    pub fn change_parameter_type(c: i64) -> i32;
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg = "cfail2")]
#[rustc_clean(cfg = "cfail3")]
#[rustc_clean(cfg = "cfail5")]
#[rustc_clean(cfg = "cfail6")]
extern "C" {
    pub fn change_parameter_type(c: i32) -> i32;
}

// Change return type ----------------------------------------------------------
#[cfg(any(cfail1,cfail4))]
extern "C" {
    pub fn change_return_type(c: i32) -> i32;
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg = "cfail2")]
#[rustc_clean(cfg = "cfail3")]
#[rustc_clean(cfg = "cfail5")]
#[rustc_clean(cfg = "cfail6")]
extern "C" {
    pub fn change_return_type(c: i32) -> i8 ;
}

// Add parameter ---------------------------------------------------------------
#[cfg(any(cfail1,cfail4))]
extern "C" {
    pub fn add_parameter(c: i32        ) -> i32;
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg = "cfail2")]
#[rustc_clean(cfg = "cfail3")]
#[rustc_clean(cfg = "cfail5")]
#[rustc_clean(cfg = "cfail6")]
extern "C" {
    pub fn add_parameter(c: i32, d: i32) -> i32;
}

// Add return type -------------------------------------------------------------
#[cfg(any(cfail1,cfail4))]
extern "C" {
    pub fn add_return_type(c: i32)       ;
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg = "cfail2")]
#[rustc_clean(cfg = "cfail3")]
#[rustc_clean(cfg = "cfail5")]
#[rustc_clean(cfg = "cfail6")]
extern "C" {
    pub fn add_return_type(c: i32) -> i32;
}

// Make function variadic ------------------------------------------------------
#[cfg(any(cfail1,cfail4))]
extern "C" {
    pub fn make_function_variadic(c: i32     );
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg = "cfail2")]
#[rustc_clean(cfg = "cfail3")]
#[rustc_clean(cfg = "cfail5")]
#[rustc_clean(cfg = "cfail6")]
extern "C" {
    pub fn make_function_variadic(c: i32, ...);
}

// Change calling convention ---------------------------------------------------
#[cfg(any(cfail1,cfail4))]
extern "C" {
    pub fn change_calling_convention(c: i32);
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg = "cfail2", except = "hir_owner")]
#[rustc_clean(cfg = "cfail3")]
#[rustc_clean(cfg = "cfail5", except = "hir_owner")]
#[rustc_clean(cfg = "cfail6")]
extern "rust-call" {
    pub fn change_calling_convention(c: i32);
}

// Make function public --------------------------------------------------------
#[cfg(any(cfail1,cfail4))]
extern "C" {
    fn     make_function_public(c: i32);
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg = "cfail2")]
#[rustc_clean(cfg = "cfail3")]
#[rustc_clean(cfg = "cfail5")]
#[rustc_clean(cfg = "cfail6")]
extern "C" {
    pub fn make_function_public(c: i32);
}

// Add function ----------------------------------------------------------------
#[cfg(any(cfail1,cfail4))]
extern "C" {
    pub fn add_function1(c: i32);
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg = "cfail2", except = "hir_owner")]
#[rustc_clean(cfg = "cfail3")]
#[rustc_clean(cfg = "cfail5", except = "hir_owner")]
#[rustc_clean(cfg = "cfail6")]
extern "C" {
    pub fn add_function1(c: i32);
    pub fn add_function2();
}

// Change link-name ------------------------------------------------------------
#[cfg(any(cfail1,cfail4))]
#[link(name = "foo")]
extern "C" {
    pub fn change_link_name(c: i32);
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg = "cfail2")]
#[rustc_clean(cfg = "cfail3")]
#[rustc_clean(cfg = "cfail5")]
#[rustc_clean(cfg = "cfail6")]
#[link(name = "bar")]
extern "C" {
    pub fn change_link_name(c: i32);
}

type c_i32 = i32;
type c_i64 = i64;

// Indirectly change parameter type --------------------------------------------
mod indirectly_change_parameter_type {
    #[cfg(any(cfail1,cfail4))]
    use super::c_i32 as c_int;
    #[cfg(not(any(cfail1,cfail4)))]
    use super::c_i64 as c_int;

    #[rustc_clean(cfg = "cfail2")]
    #[rustc_clean(cfg = "cfail3")]
    #[rustc_clean(cfg = "cfail5")]
    #[rustc_clean(cfg = "cfail6")]
    extern "C" {
        pub fn indirectly_change_parameter_type(c: c_int);
    }
}

// Indirectly change return type --------------------------------------------
mod indirectly_change_return_type {
    #[cfg(any(cfail1,cfail4))]
    use super::c_i32 as c_int;
    #[cfg(not(any(cfail1,cfail4)))]
    use super::c_i64 as c_int;

    #[rustc_clean(cfg = "cfail2")]
    #[rustc_clean(cfg = "cfail3")]
    #[rustc_clean(cfg = "cfail5")]
    #[rustc_clean(cfg = "cfail6")]
    extern "C" {
        pub fn indirectly_change_return_type() -> c_int;
    }
}
