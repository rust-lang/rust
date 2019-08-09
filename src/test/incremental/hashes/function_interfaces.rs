// This test case tests the incremental compilation hash (ICH) implementation
// for function interfaces.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

// build-pass (FIXME(62277): could be check-pass?)
// revisions: cfail1 cfail2 cfail3
// compile-flags: -Z query-dep-graph -Zincremental-ignore-spans


#![allow(warnings)]
#![feature(intrinsics)]
#![feature(linkage)]
#![feature(rustc_attrs)]
#![crate_type = "rlib"]


// Add Parameter ---------------------------------------------------------------

#[cfg(cfail1)]
pub fn add_parameter() {}

#[cfg(not(cfail1))]
#[rustc_clean(cfg = "cfail2",
              except = "Hir, HirBody, mir_built, optimized_mir, typeck_tables_of, fn_sig")]
#[rustc_clean(cfg = "cfail3")]
pub fn add_parameter(p: i32) {}


// Add Return Type -------------------------------------------------------------

#[cfg(cfail1)]
pub fn add_return_type() {}

#[cfg(not(cfail1))]
#[rustc_clean(cfg = "cfail2", except = "Hir, HirBody")]
#[rustc_clean(cfg = "cfail3")]
pub fn add_return_type() -> () {}


// Change Parameter Type -------------------------------------------------------

#[cfg(cfail1)]
pub fn type_of_parameter(p: i32) {}

#[cfg(not(cfail1))]
#[rustc_clean(cfg = "cfail2",
              except = "Hir, HirBody, mir_built, optimized_mir, typeck_tables_of, fn_sig")]
#[rustc_clean(cfg = "cfail3")]
pub fn type_of_parameter(p: i64) {}


// Change Parameter Type Reference ---------------------------------------------

#[cfg(cfail1)]
pub fn type_of_parameter_ref(p: &i32) {}

#[cfg(not(cfail1))]
#[rustc_clean(cfg = "cfail2",
              except = "Hir, HirBody, mir_built, optimized_mir, typeck_tables_of, fn_sig")]
#[rustc_clean(cfg = "cfail3")]
pub fn type_of_parameter_ref(p: &mut i32) {}


// Change Parameter Order ------------------------------------------------------

#[cfg(cfail1)]
pub fn order_of_parameters(p1: i32, p2: i64) {}

#[cfg(not(cfail1))]
#[rustc_clean(cfg = "cfail2",
              except = "Hir, HirBody, mir_built, optimized_mir, typeck_tables_of, fn_sig")]
#[rustc_clean(cfg = "cfail3")]
pub fn order_of_parameters(p2: i64, p1: i32) {}


// Unsafe ----------------------------------------------------------------------

#[cfg(cfail1)]
pub fn make_unsafe() {}

#[cfg(not(cfail1))]
#[rustc_clean(cfg = "cfail2",
              except = "Hir, HirBody, mir_built, optimized_mir, typeck_tables_of, fn_sig")]
#[rustc_clean(cfg = "cfail3")]
pub unsafe fn make_unsafe() {}


// Extern ----------------------------------------------------------------------

#[cfg(cfail1)]
pub fn make_extern() {}

#[cfg(not(cfail1))]
#[rustc_clean(cfg = "cfail2", except = "Hir, HirBody, mir_built, typeck_tables_of, fn_sig")]
#[rustc_clean(cfg = "cfail3")]
pub extern "C" fn make_extern() {}


// Extern C Extern Rust-Intrinsic ----------------------------------------------

#[cfg(cfail1)]
pub extern "C" fn make_intrinsic() {}

#[cfg(not(cfail1))]
#[rustc_clean(cfg = "cfail2", except = "Hir, HirBody, typeck_tables_of, fn_sig")]
#[rustc_clean(cfg = "cfail3")]
pub extern "rust-intrinsic" fn make_intrinsic() {}


// Type Parameter --------------------------------------------------------------

#[cfg(cfail1)]
pub fn type_parameter() {}

#[cfg(not(cfail1))]
#[rustc_clean(cfg = "cfail2",
              except = "Hir, HirBody, generics_of, type_of, predicates_of")]
#[rustc_clean(cfg = "cfail3")]
pub fn type_parameter<T>() {}


// Lifetime Parameter ----------------------------------------------------------

#[cfg(cfail1)]
pub fn lifetime_parameter() {}

#[cfg(not(cfail1))]
#[rustc_clean(cfg = "cfail2", except = "Hir, HirBody, generics_of")]
#[rustc_clean(cfg = "cfail3")]
pub fn lifetime_parameter<'a>() {}


// Trait Bound -----------------------------------------------------------------

#[cfg(cfail1)]
pub fn trait_bound<T>() {}

#[cfg(not(cfail1))]
#[rustc_clean(cfg = "cfail2", except = "Hir, HirBody, predicates_of")]
#[rustc_clean(cfg = "cfail3")]
pub fn trait_bound<T: Eq>() {}


// Builtin Bound ---------------------------------------------------------------

#[cfg(cfail1)]
pub fn builtin_bound<T>() {}

#[cfg(not(cfail1))]
#[rustc_clean(cfg = "cfail2", except = "Hir, HirBody, predicates_of")]
#[rustc_clean(cfg = "cfail3")]
pub fn builtin_bound<T: Send>() {}


// Lifetime Bound --------------------------------------------------------------

#[cfg(cfail1)]
pub fn lifetime_bound<'a, T>() {}

#[cfg(not(cfail1))]
#[rustc_clean(cfg = "cfail2",
              except = "Hir, HirBody, generics_of, type_of, predicates_of")]
#[rustc_clean(cfg = "cfail3")]
pub fn lifetime_bound<'a, T: 'a>() {}


// Second Trait Bound ----------------------------------------------------------

#[cfg(cfail1)]
pub fn second_trait_bound<T: Eq>() {}

#[cfg(not(cfail1))]
#[rustc_clean(cfg = "cfail2", except = "Hir, HirBody, predicates_of")]
#[rustc_clean(cfg = "cfail3")]
pub fn second_trait_bound<T: Eq + Clone>() {}


// Second Builtin Bound --------------------------------------------------------

#[cfg(cfail1)]
pub fn second_builtin_bound<T: Send>() {}

#[cfg(not(cfail1))]
#[rustc_clean(cfg = "cfail2", except = "Hir, HirBody, predicates_of")]
#[rustc_clean(cfg = "cfail3")]
pub fn second_builtin_bound<T: Send + Sized>() {}


// Second Lifetime Bound -------------------------------------------------------

#[cfg(cfail1)]
pub fn second_lifetime_bound<'a, 'b, T: 'a>() {}

#[cfg(not(cfail1))]
#[rustc_clean(cfg = "cfail2",
              except = "Hir, HirBody, generics_of, type_of, predicates_of")]
#[rustc_clean(cfg = "cfail3")]
pub fn second_lifetime_bound<'a, 'b, T: 'a + 'b>() {}


// Inline ----------------------------------------------------------------------

#[cfg(cfail1)]
pub fn inline() {}

#[cfg(not(cfail1))]
#[rustc_clean(cfg = "cfail2", except = "Hir, HirBody")]
#[rustc_clean(cfg = "cfail3")]
#[inline]
pub fn inline() {}


// Inline Never ----------------------------------------------------------------

#[cfg(cfail1)]
#[inline(always)]
pub fn inline_never() {}

#[cfg(not(cfail1))]
#[rustc_clean(cfg = "cfail2", except = "Hir, HirBody")]
#[rustc_clean(cfg = "cfail3")]
#[inline(never)]
pub fn inline_never() {}


// No Mangle -------------------------------------------------------------------

#[cfg(cfail1)]
pub fn no_mangle() {}

#[cfg(not(cfail1))]
#[rustc_clean(cfg = "cfail2", except = "Hir, HirBody")]
#[rustc_clean(cfg = "cfail3")]
#[no_mangle]
pub fn no_mangle() {}


// Linkage ---------------------------------------------------------------------

#[cfg(cfail1)]
pub fn linkage() {}

#[cfg(not(cfail1))]
#[rustc_clean(cfg = "cfail2", except = "Hir, HirBody")]
#[rustc_clean(cfg = "cfail3")]
#[linkage = "weak_odr"]
pub fn linkage() {}


// Return Impl Trait -----------------------------------------------------------

#[cfg(cfail1)]
pub fn return_impl_trait() -> i32 {
    0
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg = "cfail2", except = "Hir, HirBody, typeck_tables_of, fn_sig")]
#[rustc_clean(cfg = "cfail3")]
pub fn return_impl_trait() -> impl Clone {
    0
}


// Change Return Impl Trait ----------------------------------------------------

#[cfg(cfail1)]
pub fn change_return_impl_trait() -> impl Clone {
    0u32
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg = "cfail2")]
#[rustc_clean(cfg = "cfail3")]
pub fn change_return_impl_trait() -> impl Copy {
    0u32
}


// Change Return Type Indirectly -----------------------------------------------

pub struct ReferencedType1;
pub struct ReferencedType2;

pub mod change_return_type_indirectly {
    #[cfg(cfail1)]
    use super::ReferencedType1 as ReturnType;
    #[cfg(not(cfail1))]
    use super::ReferencedType2 as ReturnType;

    #[rustc_clean(cfg = "cfail2",
                  except = "Hir, HirBody, mir_built, optimized_mir, typeck_tables_of, fn_sig")]
    #[rustc_clean(cfg = "cfail3")]
    pub fn indirect_return_type() -> ReturnType {
        ReturnType {}
    }
}


// Change Parameter Type Indirectly --------------------------------------------

pub mod change_parameter_type_indirectly {
    #[cfg(cfail1)]
    use super::ReferencedType1 as ParameterType;
    #[cfg(not(cfail1))]
    use super::ReferencedType2 as ParameterType;

    #[rustc_clean(cfg = "cfail2",
                  except = "Hir, HirBody, mir_built, optimized_mir, typeck_tables_of, fn_sig")]
    #[rustc_clean(cfg = "cfail3")]
    pub fn indirect_parameter_type(p: ParameterType) {}
}


// Change Trait Bound Indirectly -----------------------------------------------

pub trait ReferencedTrait1 {}
pub trait ReferencedTrait2 {}

pub mod change_trait_bound_indirectly {
    #[cfg(cfail1)]
    use super::ReferencedTrait1 as Trait;
    #[cfg(not(cfail1))]
    use super::ReferencedTrait2 as Trait;

    #[rustc_clean(cfg = "cfail2", except = "Hir, HirBody, predicates_of")]
    #[rustc_clean(cfg = "cfail3")]
    pub fn indirect_trait_bound<T: Trait>(p: T) {}
}


// Change Trait Bound Indirectly In Where Clause -------------------------------

pub mod change_trait_bound_indirectly_in_where_clause {
    #[cfg(cfail1)]
    use super::ReferencedTrait1 as Trait;
    #[cfg(not(cfail1))]
    use super::ReferencedTrait2 as Trait;

    #[rustc_clean(cfg = "cfail2", except = "Hir, HirBody, predicates_of")]
    #[rustc_clean(cfg = "cfail3")]
    pub fn indirect_trait_bound_where<T>(p: T)
    where
        T: Trait,
    {
    }
}
