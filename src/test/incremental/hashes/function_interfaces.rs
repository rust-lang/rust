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
// for function interfaces.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

// must-compile-successfully
// revisions: cfail1 cfail2 cfail3
// compile-flags: -Z query-dep-graph


#![allow(warnings)]
#![feature(conservative_impl_trait)]
#![feature(intrinsics)]
#![feature(linkage)]
#![feature(rustc_attrs)]
#![crate_type="rlib"]


// Add Parameter ---------------------------------------------------------------

#[cfg(cfail1)]
fn add_parameter() {}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn add_parameter(p: i32) {}


// Add Return Type -------------------------------------------------------------

#[cfg(cfail1)]
fn add_return_type() {}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn add_return_type() -> () {}


// Change Parameter Type -------------------------------------------------------

#[cfg(cfail1)]
fn type_of_parameter(p: i32) {}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn type_of_parameter(p: i64) {}


// Change Parameter Type Reference ---------------------------------------------

#[cfg(cfail1)]
fn type_of_parameter_ref(p: &i32) {}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn type_of_parameter_ref(p: &mut i32) {}


// Change Parameter Order ------------------------------------------------------

#[cfg(cfail1)]
fn order_of_parameters(p1: i32, p2: i64) {}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn order_of_parameters(p2: i64, p1: i32) {}


// Unsafe ----------------------------------------------------------------------

#[cfg(cfail1)]
fn make_unsafe() {}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
unsafe fn make_unsafe() {}


// Extern ----------------------------------------------------------------------

#[cfg(cfail1)]
fn make_extern() {}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
extern fn make_extern() {}


// Extern C Extern Rust-Intrinsic ----------------------------------------------

#[cfg(cfail1)]
extern "C" fn make_intrinsic() {}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
extern "rust-intrinsic" fn make_intrinsic() {}


// Type Parameter --------------------------------------------------------------

#[cfg(cfail1)]
fn type_parameter() {}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn type_parameter<T>() {}


// Lifetime Parameter ----------------------------------------------------------

#[cfg(cfail1)]
fn lifetime_parameter() {}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn lifetime_parameter<'a>() {}


// Trait Bound -----------------------------------------------------------------

#[cfg(cfail1)]
fn trait_bound<T>() {}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn trait_bound<T: Eq>() {}


// Builtin Bound ---------------------------------------------------------------

#[cfg(cfail1)]
fn builtin_bound<T>() {}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn builtin_bound<T: Send>() {}


// Lifetime Bound --------------------------------------------------------------

#[cfg(cfail1)]
fn lifetime_bound<'a, T>() {}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn lifetime_bound<'a, T: 'a>() {}


// Second Trait Bound ----------------------------------------------------------

#[cfg(cfail1)]
fn second_trait_bound<T: Eq>() {}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn second_trait_bound<T: Eq + Clone>() {}


// Second Builtin Bound --------------------------------------------------------

#[cfg(cfail1)]
fn second_builtin_bound<T: Send>() {}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn second_builtin_bound<T: Send + Sized>() {}


// Second Lifetime Bound -------------------------------------------------------

#[cfg(cfail1)]
fn second_lifetime_bound<'a, 'b, T: 'a>() {}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn second_lifetime_bound<'a, 'b, T: 'a + 'b>() {}


// Inline ----------------------------------------------------------------------

#[cfg(cfail1)]
fn inline() {}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
#[inline]
fn inline() {}


// Inline Never ----------------------------------------------------------------

#[cfg(cfail1)]
#[inline(always)]
fn inline_never() {}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
#[inline(never)]
fn inline_never() {}


// No Mangle -------------------------------------------------------------------

#[cfg(cfail1)]
fn no_mangle() {}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
#[no_mangle]
fn no_mangle() {}


// Linkage ---------------------------------------------------------------------

#[cfg(cfail1)]
fn linkage() {}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
#[linkage="weak_odr"]
fn linkage() {}


// Return Impl Trait -----------------------------------------------------------

#[cfg(cfail1)]
fn return_impl_trait() -> i32 {
    0
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn return_impl_trait() -> impl Clone {
    0
}


// Change Return Impl Trait ----------------------------------------------------

#[cfg(cfail1)]
fn change_return_impl_trait() -> impl Clone {
    0
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn change_return_impl_trait() -> impl Copy {
    0
}


// Change Return Type Indirectly -----------------------------------------------

struct ReferencedType1;
struct ReferencedType2;

mod change_return_type_indirectly {
    #[cfg(cfail1)]
    use super::ReferencedType1 as ReturnType;
    #[cfg(not(cfail1))]
    use super::ReferencedType2 as ReturnType;

    #[rustc_dirty(label="Hir", cfg="cfail2")]
    #[rustc_clean(label="Hir", cfg="cfail3")]
    #[rustc_metadata_dirty(cfg="cfail2")]
    #[rustc_metadata_clean(cfg="cfail3")]
    fn indirect_return_type() -> ReturnType {
        ReturnType {}
    }
}


// Change Parameter Type Indirectly --------------------------------------------

mod change_parameter_type_indirectly {
    #[cfg(cfail1)]
    use super::ReferencedType1 as ParameterType;
    #[cfg(not(cfail1))]
    use super::ReferencedType2 as ParameterType;

    #[rustc_dirty(label="Hir", cfg="cfail2")]
    #[rustc_clean(label="Hir", cfg="cfail3")]
    #[rustc_metadata_dirty(cfg="cfail2")]
    #[rustc_metadata_clean(cfg="cfail3")]
    fn indirect_parameter_type(p: ParameterType) {}
}


// Change Trait Bound Indirectly -----------------------------------------------

trait ReferencedTrait1 {}
trait ReferencedTrait2 {}

mod change_trait_bound_indirectly {
    #[cfg(cfail1)]
    use super::ReferencedTrait1 as Trait;
    #[cfg(not(cfail1))]
    use super::ReferencedTrait2 as Trait;

    #[rustc_dirty(label="Hir", cfg="cfail2")]
    #[rustc_clean(label="Hir", cfg="cfail3")]
    #[rustc_metadata_dirty(cfg="cfail2")]
    #[rustc_metadata_clean(cfg="cfail3")]
    fn indirect_trait_bound<T: Trait>(p: T) {}
}


// Change Trait Bound Indirectly In Where Clause -------------------------------

mod change_trait_bound_indirectly_in_where_clause {
    #[cfg(cfail1)]
    use super::ReferencedTrait1 as Trait;
    #[cfg(not(cfail1))]
    use super::ReferencedTrait2 as Trait;

    #[rustc_dirty(label="Hir", cfg="cfail2")]
    #[rustc_clean(label="Hir", cfg="cfail3")]
    #[rustc_metadata_dirty(cfg="cfail2")]
    #[rustc_metadata_clean(cfg="cfail3")]
    fn indirect_trait_bound_where<T>(p: T) where T: Trait {}
}
