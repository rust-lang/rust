// This test case tests the incremental compilation hash (ICH) implementation
// for let expressions.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

// build-pass (FIXME(62277): could be check-pass?)
// revisions: cfail1 cfail2 cfail3
// compile-flags: -Z query-dep-graph -Zincremental-ignore-spans


#![allow(warnings)]
#![feature(rustc_attrs)]
#![crate_type="rlib"]

pub struct Foo;

// Change Method Name -----------------------------------------------------------
#[cfg(cfail1)]
impl Foo {
    pub fn method_name() { }
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="Hir,HirBody,associated_item_def_ids")]
#[rustc_clean(cfg="cfail3")]
impl Foo {
    #[rustc_clean(cfg="cfail3")]
    pub fn method_name2() { }
}

// Change Method Body -----------------------------------------------------------
//
// This should affect the method itself, but not the impl.
#[cfg(cfail1)]
impl Foo {
    pub fn method_body() { }
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
impl Foo {
    #[rustc_clean(cfg="cfail2", except="HirBody,optimized_mir,mir_built,typeck_tables_of")]
    #[rustc_clean(cfg="cfail3")]
    pub fn method_body() {
        println!("Hello, world!");
    }
}


// Change Method Body (inlined) ------------------------------------------------
//
// This should affect the method itself, but not the impl.
#[cfg(cfail1)]
impl Foo {
    #[inline]
    pub fn method_body_inlined() { }
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
impl Foo {
    #[rustc_clean(cfg="cfail2", except="HirBody,optimized_mir,mir_built,typeck_tables_of")]
    #[rustc_clean(cfg="cfail3")]
    #[inline]
    pub fn method_body_inlined() {
        println!("Hello, world!");
    }
}


// Change Method Privacy -------------------------------------------------------
#[cfg(cfail1)]
impl Foo {
    pub fn method_privacy() { }
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="Hir,HirBody")]
#[rustc_clean(cfg="cfail3")]
impl Foo {
    #[rustc_clean(cfg="cfail2", except="associated_item,Hir,HirBody")]
    #[rustc_clean(cfg="cfail3")]
    fn method_privacy() { }
}

// Change Method Selfness -----------------------------------------------------------
#[cfg(cfail1)]
impl Foo {
    pub fn method_selfness() { }
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="Hir,HirBody")]
#[rustc_clean(cfg="cfail3")]
impl Foo {
    #[rustc_dirty(cfg="cfail2", except="type_of,predicates_of")]
    #[rustc_clean(cfg="cfail3")]
    pub fn method_selfness(&self) { }
}

// Change Method Selfmutness ---------------------------------------------------
#[cfg(cfail1)]
impl Foo {
    pub fn method_selfmutness(&self) { }
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
impl Foo {
    #[rustc_clean(
        cfg="cfail2",
        except="Hir,HirBody,fn_sig,typeck_tables_of,optimized_mir,mir_built"
    )]
    #[rustc_clean(cfg="cfail3")]
    pub fn method_selfmutness(&mut self) { }
}



// Add Method To Impl ----------------------------------------------------------
#[cfg(cfail1)]
impl Foo {
    pub fn add_method_to_impl1(&self) { }
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="Hir,HirBody,associated_item_def_ids")]
#[rustc_clean(cfg="cfail3")]
impl Foo {
    #[rustc_clean(cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    pub fn add_method_to_impl1(&self) { }

    #[rustc_clean(cfg="cfail3")]
    pub fn add_method_to_impl2(&self) { }
}



// Add Method Parameter --------------------------------------------------------
#[cfg(cfail1)]
impl Foo {
    pub fn add_method_parameter(&self) { }
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
impl Foo {
    #[rustc_clean(
        cfg="cfail2",
        except="Hir,HirBody,fn_sig,typeck_tables_of,optimized_mir,mir_built"
    )]
    #[rustc_clean(cfg="cfail3")]
    pub fn add_method_parameter(&self, _: i32) { }
}



// Change Method Parameter Name ------------------------------------------------
#[cfg(cfail1)]
impl Foo {
    pub fn change_method_parameter_name(&self, a: i64) { }
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
impl Foo {
    #[rustc_clean(cfg="cfail2", except="HirBody,optimized_mir,mir_built")]
    #[rustc_clean(cfg="cfail3")]
    pub fn change_method_parameter_name(&self, b: i64) { }
}



// Change Method Return Type ---------------------------------------------------
#[cfg(cfail1)]
impl Foo {
    pub fn change_method_return_type(&self) -> u16 { 0 }
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
impl Foo {
    #[rustc_clean(
        cfg="cfail2",
        except="Hir,HirBody,fn_sig,optimized_mir,mir_built,typeck_tables_of")]
    #[rustc_clean(cfg="cfail3")]
    pub fn change_method_return_type(&self) -> u8 { 0 }
}



// Make Method #[inline] -------------------------------------------------------
#[cfg(cfail1)]
impl Foo {
    pub fn make_method_inline(&self) -> u8 { 0 }
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
impl Foo {
    #[rustc_clean(cfg="cfail2", except="Hir,HirBody")]
    #[rustc_clean(cfg="cfail3")]
    #[inline]
    pub fn make_method_inline(&self) -> u8 { 0 }
}



//  Change order of parameters -------------------------------------------------
#[cfg(cfail1)]
impl Foo {
    pub fn change_method_parameter_order(&self, a: i64, b: i64) { }
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
impl Foo {
    #[rustc_clean(cfg="cfail2", except="HirBody,optimized_mir,mir_built")]
    #[rustc_clean(cfg="cfail3")]
    pub fn change_method_parameter_order(&self, b: i64, a: i64) { }
}



// Make method unsafe ----------------------------------------------------------
#[cfg(cfail1)]
impl Foo {
    pub fn make_method_unsafe(&self) { }
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
impl Foo {
    #[rustc_clean(
        cfg="cfail2",
        except="Hir,HirBody,fn_sig,typeck_tables_of,optimized_mir,mir_built"
    )]
    #[rustc_clean(cfg="cfail3")]
    pub unsafe fn make_method_unsafe(&self) { }
}



// Make method extern ----------------------------------------------------------
#[cfg(cfail1)]
impl Foo {
    pub fn make_method_extern(&self) { }
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
impl Foo {
    #[rustc_clean(cfg="cfail2", except="Hir,HirBody,mir_built,fn_sig,typeck_tables_of")]
    #[rustc_clean(cfg="cfail3")]
    pub extern fn make_method_extern(&self) { }
}



// Change method calling convention --------------------------------------------
#[cfg(cfail1)]
impl Foo {
    pub extern "C" fn change_method_calling_convention(&self) { }
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
impl Foo {
    #[rustc_clean(cfg="cfail2", except="Hir,HirBody,fn_sig,typeck_tables_of")]
    #[rustc_clean(cfg="cfail3")]
    pub extern "system" fn change_method_calling_convention(&self) { }
}



// Add Lifetime Parameter to Method --------------------------------------------
#[cfg(cfail1)]
impl Foo {
    pub fn add_lifetime_parameter_to_method(&self) { }
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
impl Foo {
    // Warning: Note that `typeck_tables_of` are coming up clean here.
    // The addition or removal of lifetime parameters that don't
    // appear in the arguments or fn body in any way does not, in
    // fact, affect the `typeck_tables_of` in any semantic way (at least
    // as of this writing). **However,** altering the order of
    // lowering **can** cause it appear to affect the `typeck_tables_of`:
    // if we lower generics before the body, then the `HirId` for
    // things in the body will be affected. So if you start to see
    // `typeck_tables_of` appear dirty, that might be the cause. -nmatsakis
    #[rustc_clean(cfg="cfail2", except="Hir,HirBody")]
    #[rustc_clean(cfg="cfail3")]
    pub fn add_lifetime_parameter_to_method<'a>(&self) { }
}



// Add Type Parameter To Method ------------------------------------------------
#[cfg(cfail1)]
impl Foo {
    pub fn add_type_parameter_to_method(&self) { }
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
impl Foo {
    // Warning: Note that `typeck_tables_of` are coming up clean here.
    // The addition or removal of type parameters that don't appear in
    // the arguments or fn body in any way does not, in fact, affect
    // the `typeck_tables_of` in any semantic way (at least as of this
    // writing). **However,** altering the order of lowering **can**
    // cause it appear to affect the `typeck_tables_of`: if we lower
    // generics before the body, then the `HirId` for things in the
    // body will be affected. So if you start to see `typeck_tables_of`
    // appear dirty, that might be the cause. -nmatsakis
    #[rustc_clean(
        cfg="cfail2",
        except="Hir,HirBody,generics_of,predicates_of,type_of",
    )]
    #[rustc_clean(cfg="cfail3")]
    pub fn add_type_parameter_to_method<T>(&self) { }
}



// Add Lifetime Bound to Lifetime Parameter of Method --------------------------
#[cfg(cfail1)]
impl Foo {
    pub fn add_lifetime_bound_to_lifetime_param_of_method<'a, 'b>(&self) { }
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
impl Foo {
    #[rustc_clean(
        cfg="cfail2",
        except="Hir,HirBody,generics_of,predicates_of,type_of,typeck_tables_of"
    )]
    #[rustc_clean(cfg="cfail3")]
    pub fn add_lifetime_bound_to_lifetime_param_of_method<'a, 'b: 'a>(&self) { }
}



// Add Lifetime Bound to Type Parameter of Method ------------------------------
#[cfg(cfail1)]
impl Foo {
    pub fn add_lifetime_bound_to_type_param_of_method<'a, T>(&self) { }
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
impl Foo {
    // Warning: Note that `typeck_tables_of` are coming up clean here.
    // The addition or removal of bounds that don't appear in the
    // arguments or fn body in any way does not, in fact, affect the
    // `typeck_tables_of` in any semantic way (at least as of this
    // writing). **However,** altering the order of lowering **can**
    // cause it appear to affect the `typeck_tables_of`: if we lower
    // generics before the body, then the `HirId` for things in the
    // body will be affected. So if you start to see `typeck_tables_of`
    // appear dirty, that might be the cause. -nmatsakis
    #[rustc_clean(cfg="cfail2", except="Hir,HirBody,generics_of,predicates_of,\
                                        type_of")]
    #[rustc_clean(cfg="cfail3")]
    pub fn add_lifetime_bound_to_type_param_of_method<'a, T: 'a>(&self) { }
}



// Add Trait Bound to Type Parameter of Method ------------------------------
#[cfg(cfail1)]
impl Foo {
    pub fn add_trait_bound_to_type_param_of_method<T>(&self) { }
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
impl Foo {
    // Warning: Note that `typeck_tables_of` are coming up clean here.
    // The addition or removal of bounds that don't appear in the
    // arguments or fn body in any way does not, in fact, affect the
    // `typeck_tables_of` in any semantic way (at least as of this
    // writing). **However,** altering the order of lowering **can**
    // cause it appear to affect the `typeck_tables_of`: if we lower
    // generics before the body, then the `HirId` for things in the
    // body will be affected. So if you start to see `typeck_tables_of`
    // appear dirty, that might be the cause. -nmatsakis
    #[rustc_clean(cfg="cfail2", except="Hir,HirBody,predicates_of")]
    #[rustc_clean(cfg="cfail3")]
    pub fn add_trait_bound_to_type_param_of_method<T: Clone>(&self) { }
}



// Add #[no_mangle] to Method --------------------------------------------------
#[cfg(cfail1)]
impl Foo {
    pub fn add_no_mangle_to_method(&self) { }
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
impl Foo {
    #[rustc_clean(cfg="cfail2", except="Hir,HirBody")]
    #[rustc_clean(cfg="cfail3")]
    #[no_mangle]
    pub fn add_no_mangle_to_method(&self) { }
}



struct Bar<T>(T);

// Add Type Parameter To Impl --------------------------------------------------
#[cfg(cfail1)]
impl Bar<u32> {
    pub fn add_type_parameter_to_impl(&self) { }
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="Hir,HirBody,generics_of")]
#[rustc_clean(cfg="cfail3")]
impl<T> Bar<T> {
    #[rustc_clean(
        cfg="cfail2",
        except="generics_of,fn_sig,typeck_tables_of,type_of,optimized_mir,mir_built"
    )]
    #[rustc_clean(cfg="cfail3")]
    pub fn add_type_parameter_to_impl(&self) { }
}



// Change Self Type of Impl ----------------------------------------------------
#[cfg(cfail1)]
impl Bar<u32> {
    pub fn change_impl_self_type(&self) { }
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="Hir,HirBody")]
#[rustc_clean(cfg="cfail3")]
impl Bar<u64> {
    #[rustc_clean(cfg="cfail2", except="fn_sig,optimized_mir,mir_built,typeck_tables_of")]
    #[rustc_clean(cfg="cfail3")]
    pub fn change_impl_self_type(&self) { }
}



// Add Lifetime Bound to Impl --------------------------------------------------
#[cfg(cfail1)]
impl<T> Bar<T> {
    pub fn add_lifetime_bound_to_impl_parameter(&self) { }
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="Hir,HirBody")]
#[rustc_clean(cfg="cfail3")]
impl<T: 'static> Bar<T> {
    #[rustc_clean(cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    pub fn add_lifetime_bound_to_impl_parameter(&self) { }
}



// Add Trait Bound to Impl Parameter -------------------------------------------
#[cfg(cfail1)]
impl<T> Bar<T> {
    pub fn add_trait_bound_to_impl_parameter(&self) { }
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="Hir,HirBody")]
#[rustc_clean(cfg="cfail3")]
impl<T: Clone> Bar<T> {
    #[rustc_clean(cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    pub fn add_trait_bound_to_impl_parameter(&self) { }
}


// Force instantiation of some fns so we can check their hash.
pub fn instantiation_root() {
    Foo::method_privacy();

    #[cfg(cfail1)]
    {
        Bar(0u32).change_impl_self_type();
    }

    #[cfg(not(cfail1))]
    {
        Bar(0u64).change_impl_self_type();
    }
}
