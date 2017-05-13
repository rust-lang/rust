// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// This test case tests the incremental compilation hash (ICH) implementation
// for `extern` modules.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

// must-compile-successfully
// revisions: cfail1 cfail2 cfail3
// compile-flags: -Z query-dep-graph

#![allow(warnings)]
#![feature(rustc_attrs)]
#![feature(unboxed_closures)]
#![feature(link_args)]
#![crate_type="rlib"]


// Change function name --------------------------------------------------------
#[cfg(cfail1)]
extern {
    pub fn change_function_name1(c: i64) -> i32;
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
extern {
    pub fn change_function_name2(c: i64) -> i32;
}



// Change parameter name -------------------------------------------------------
#[cfg(cfail1)]
extern {
    pub fn change_parameter_name(c: i64) -> i32;
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
extern {
    pub fn change_parameter_name(d: i64) -> i32;
}



// Change parameter type -------------------------------------------------------
#[cfg(cfail1)]
extern {
    pub fn change_parameter_type(c: i64) -> i32;
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
extern {
    pub fn change_parameter_type(c: i32) -> i32;
}



// Change return type ----------------------------------------------------------
#[cfg(cfail1)]
extern {
    pub fn change_return_type(c: i32) -> i32;
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
extern {
    pub fn change_return_type(c: i32) -> i8;
}



// Add parameter ---------------------------------------------------------------
#[cfg(cfail1)]
extern {
    pub fn add_parameter(c: i32) -> i32;
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
extern {
    pub fn add_parameter(c: i32, d: i32) -> i32;
}



// Add return type -------------------------------------------------------------
#[cfg(cfail1)]
extern {
    pub fn add_return_type(c: i32);
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
extern {
    pub fn add_return_type(c: i32) -> i32;
}



// Make function variadic ------------------------------------------------------
#[cfg(cfail1)]
extern {
    pub fn make_function_variadic(c: i32);
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
extern {
    pub fn make_function_variadic(c: i32, ...);
}



// Change calling convention ---------------------------------------------------
#[cfg(cfail1)]
extern "C" {
    pub fn change_calling_convention(c: i32);
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
extern "rust-call" {
    pub fn change_calling_convention(c: i32);
}



// Make function public --------------------------------------------------------
#[cfg(cfail1)]
extern {
    fn make_function_public(c: i32);
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
extern {
    pub fn make_function_public(c: i32);
}



// Add function ----------------------------------------------------------------
#[cfg(cfail1)]
extern {
    pub fn add_function1(c: i32);
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
extern {
    pub fn add_function1(c: i32);
    pub fn add_function2();
}



// Change link-args ------------------------------------------------------------
#[cfg(cfail1)]
#[link_args = "-foo -bar"]
extern {
    pub fn change_link_args(c: i32);
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
#[link_args = "-foo -bar -baz"]
extern {
    pub fn change_link_args(c: i32);
}



// Change link-name ------------------------------------------------------------
#[cfg(cfail1)]
#[link(name = "foo")]
extern {
    pub fn change_link_name(c: i32);
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
#[link(name = "bar")]
extern {
    pub fn change_link_name(c: i32);
}

type c_i32 = i32;
type c_i64 = i64;

// Indirectly change parameter type --------------------------------------------
mod indirectly_change_parameter_type {
    #[cfg(cfail1)]
    use super::c_i32 as c_int;
    #[cfg(not(cfail1))]
    use super::c_i64 as c_int;

    #[rustc_dirty(label="Hir", cfg="cfail2")]
    #[rustc_clean(label="Hir", cfg="cfail3")]
    #[rustc_metadata_dirty(cfg="cfail2")]
    #[rustc_metadata_clean(cfg="cfail3")]
    extern {
        pub fn indirectly_change_parameter_type(c: c_int);
    }
}



// Indirectly change return type --------------------------------------------
mod indirectly_change_return_type {
    #[cfg(cfail1)]
    use super::c_i32 as c_int;
    #[cfg(not(cfail1))]
    use super::c_i64 as c_int;

    #[rustc_dirty(label="Hir", cfg="cfail2")]
    #[rustc_clean(label="Hir", cfg="cfail3")]
    #[rustc_metadata_dirty(cfg="cfail2")]
    #[rustc_metadata_clean(cfg="cfail3")]
    extern {
        pub fn indirectly_change_return_type() -> c_int;
    }
}
