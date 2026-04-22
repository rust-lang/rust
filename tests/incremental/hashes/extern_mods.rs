// This test case tests the incremental compilation hash (ICH) implementation
// for `extern` modules.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

//@ revisions: bpass1 bpass2 bpass3 bpass4 bpass5 bpass6
//@ compile-flags: -Z query-dep-graph -O
//@ [bpass1]compile-flags: -Zincremental-ignore-spans
//@ [bpass2]compile-flags: -Zincremental-ignore-spans
//@ [bpass3]compile-flags: -Zincremental-ignore-spans
//@ ignore-backends: gcc
// FIXME(#62277): could be check-pass?

#![allow(warnings)]
#![feature(rustc_attrs)]
#![feature(unboxed_closures)]
#![crate_type = "rlib"]

// Change function name --------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
extern "C" {
    pub fn change_function_name1(c: i64) -> i32;
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg = "bpass2", except = "opt_hir_owner_nodes")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(cfg = "bpass5", except = "opt_hir_owner_nodes")]
#[rustc_clean(cfg = "bpass6")]
extern "C" {
    pub fn change_function_name2(c: i64) -> i32;
}

// Change parameter name -------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
extern "C" {
    pub fn change_parameter_name(c: i64) -> i32;
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg = "bpass2")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(cfg = "bpass5")]
#[rustc_clean(cfg = "bpass6")]
extern "C" {
    pub fn change_parameter_name(d: i64) -> i32;
}

// Change parameter type -------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
extern "C" {
    pub fn change_parameter_type(c: i64) -> i32;
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg = "bpass2")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(cfg = "bpass5")]
#[rustc_clean(cfg = "bpass6")]
extern "C" {
    pub fn change_parameter_type(c: i32) -> i32;
}

// Change return type ----------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
extern "C" {
    pub fn change_return_type(c: i32) -> i32;
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg = "bpass2")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(cfg = "bpass5")]
#[rustc_clean(cfg = "bpass6")]
extern "C" {
    pub fn change_return_type(c: i32) -> i8 ;
}

// Add parameter ---------------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
extern "C" {
    pub fn add_parameter(c: i32        ) -> i32;
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg = "bpass2")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(cfg = "bpass5")]
#[rustc_clean(cfg = "bpass6")]
extern "C" {
    pub fn add_parameter(c: i32, d: i32) -> i32;
}

// Add return type -------------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
extern "C" {
    pub fn add_return_type(c: i32)       ;
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg = "bpass2")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(cfg = "bpass5")]
#[rustc_clean(cfg = "bpass6")]
extern "C" {
    pub fn add_return_type(c: i32) -> i32;
}

// Make function variadic ------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
extern "C" {
    pub fn make_function_variadic(c: i32     );
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg = "bpass2")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(cfg = "bpass5")]
#[rustc_clean(cfg = "bpass6")]
extern "C" {
    pub fn make_function_variadic(c: i32, ...);
}

// Change calling convention ---------------------------------------------------
#[cfg(any(bpass1,bpass4))]
extern "C" {
    pub fn change_calling_convention(c: (i32,));
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg = "bpass2", except = "opt_hir_owner_nodes")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(cfg = "bpass5", except = "opt_hir_owner_nodes")]
#[rustc_clean(cfg = "bpass6")]
extern "rust-call" {
    pub fn change_calling_convention(c: (i32,));
}

// Make function public --------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
extern "C" {
    fn     make_function_public(c: i32);
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg = "bpass2")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(cfg = "bpass5")]
#[rustc_clean(cfg = "bpass6")]
extern "C" {
    pub fn make_function_public(c: i32);
}

// Add function ----------------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
extern "C" {
    pub fn add_function1(c: i32);
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg = "bpass2", except = "opt_hir_owner_nodes")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(cfg = "bpass5", except = "opt_hir_owner_nodes")]
#[rustc_clean(cfg = "bpass6")]
extern "C" {
    pub fn add_function1(c: i32);
    pub fn add_function2();
}

// Change link-name ------------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
#[link(name = "foo")]
extern "C" {
    pub fn change_link_name(c: i32);
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg = "bpass2")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(cfg = "bpass5")]
#[rustc_clean(cfg = "bpass6")]
#[link(name = "bar")]
extern "C" {
    pub fn change_link_name(c: i32);
}

type c_i32 = i32;
type c_i64 = i64;

// Indirectly change parameter type --------------------------------------------
mod indirectly_change_parameter_type {
    #[cfg(any(bpass1,bpass4))]
    use super::c_i32 as c_int;
    #[cfg(not(any(bpass1,bpass4)))]
    use super::c_i64 as c_int;

    #[rustc_clean(cfg = "bpass2")]
    #[rustc_clean(cfg = "bpass3")]
    #[rustc_clean(cfg = "bpass5")]
    #[rustc_clean(cfg = "bpass6")]
    extern "C" {
        pub fn indirectly_change_parameter_type(c: c_int);
    }
}

// Indirectly change return type --------------------------------------------
mod indirectly_change_return_type {
    #[cfg(any(bpass1,bpass4))]
    use super::c_i32 as c_int;
    #[cfg(not(any(bpass1,bpass4)))]
    use super::c_i64 as c_int;

    #[rustc_clean(cfg = "bpass2")]
    #[rustc_clean(cfg = "bpass3")]
    #[rustc_clean(cfg = "bpass5")]
    #[rustc_clean(cfg = "bpass6")]
    extern "C" {
        pub fn indirectly_change_return_type() -> c_int;
    }
}
