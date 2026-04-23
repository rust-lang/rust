// This test case tests the incremental compilation hash (ICH) implementation
// for let expressions.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

//@ edition: 2024
//@ revisions: bpass1 bpass2 bpass3 bpass4 bpass5 bpass6
//@ compile-flags: -Z query-dep-graph -O
//@ [bpass1]compile-flags: -Zincremental-ignore-spans
//@ [bpass2]compile-flags: -Zincremental-ignore-spans
//@ [bpass3]compile-flags: -Zincremental-ignore-spans
//@ ignore-backends: gcc
// FIXME(#62277): could be check-pass?

#![allow(warnings)]
#![feature(rustc_attrs)]
#![crate_type="rlib"]

pub struct Foo;

// Change Method Name -----------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
impl Foo {
    pub fn method_name() { }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner,associated_item_def_ids")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="owner,associated_item_def_ids")]
#[rustc_clean(cfg="bpass6")]
impl Foo {
    #[rustc_clean(cfg="bpass3")]
    #[rustc_clean(cfg="bpass6")]
    pub fn method_name2() { }
}

// Change Method Body -----------------------------------------------------------
//
// This should affect the method itself, but not the impl.
#[cfg(any(bpass1,bpass4))]
impl Foo {
    //------------------------------------------------------------------------------------------
    //--------------------------
    //------------------------------------------------------------------------------------------
    //--------------------------
    pub fn method_body() {
        // -----------------------
    }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(except="owner", cfg="bpass5")]
#[rustc_clean(cfg="bpass6")]
impl Foo {
    #[rustc_clean(cfg="bpass2",except="owner,optimized_mir,promoted_mir,typeck_root")]
    #[rustc_clean(cfg="bpass3")]
    #[rustc_clean(cfg="bpass5",except="owner,optimized_mir,promoted_mir,typeck_root")]
    #[rustc_clean(cfg="bpass6")]
    pub fn method_body() {
        println!("Hello, world!");
    }
}


// Change Method Body (inlined) ------------------------------------------------
//
// This should affect the method itself, but not the impl.
#[cfg(any(bpass1,bpass4))]
impl Foo {
    //-----------------------------------------------------------------------------
    //--------------------------
    //-----------------------------------------------------------------------------
    //--------------------------
    #[inline]
    pub fn method_body_inlined() {
        // -----------------------
    }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(except="owner", cfg="bpass5")]
#[rustc_clean(cfg="bpass6")]
impl Foo {
    #[rustc_clean(cfg="bpass2", except="owner,optimized_mir,typeck_root")]
    #[rustc_clean(cfg="bpass3")]
    #[rustc_clean(cfg="bpass5", except="owner,optimized_mir,typeck_root")]
    #[rustc_clean(cfg="bpass6")]
    #[inline]
    pub fn method_body_inlined() {
        println!("Hello, world!");
    }
}


// Change Method Privacy -------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
impl Foo {
    //--------------------------
    //--------------------------
    //--------------------------------------------------------
    //--------------------------
    pub fn method_privacy() { }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="owner")]
#[rustc_clean(cfg="bpass6")]
impl Foo {
    #[rustc_clean(cfg="bpass2", except="owner")]
    #[rustc_clean(cfg="bpass3")]
    #[rustc_clean(cfg="bpass5", except="owner")]
    #[rustc_clean(cfg="bpass6")]
    fn     method_privacy() { }
}

// Change Method Selfness -----------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
impl Foo {
    //------------
    //---------------
    //---------------------------------------------------------------------------------------
    //
    //--------------------------
    //------------
    //---------------
    //---------------------------------------------------------------------------------------
    //
    //--------------------------
    pub fn method_selfness() { }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="owner")]
#[rustc_clean(cfg="bpass6")]
impl Foo {
    #[rustc_clean(
        cfg="bpass2",
        except="owner,fn_sig,generics_of,typeck_root,associated_item,optimized_mir",
    )]
    #[rustc_clean(cfg="bpass3")]
    #[rustc_clean(
        cfg="bpass5",
        except="owner,fn_sig,generics_of,typeck_root,associated_item,optimized_mir",
    )]
    #[rustc_clean(cfg="bpass6")]
    pub fn method_selfness(&self) { }
}

// Change Method Selfmutness ---------------------------------------------------
#[cfg(any(bpass1,bpass4))]
impl Foo {
    //------------------------------------------------------------------------------------
    //--------------------------
    //------------------------------------------------------------------------------------
    //--------------------------
    pub fn method_selfmutness(&    self) { }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(except="owner", cfg="bpass5")]
#[rustc_clean(cfg="bpass6")]
impl Foo {
    #[rustc_clean(cfg="bpass2", except="owner,fn_sig,typeck_root,optimized_mir")]
    #[rustc_clean(cfg="bpass3")]
    #[rustc_clean(cfg="bpass5", except="owner,fn_sig,typeck_root,optimized_mir")]
    #[rustc_clean(cfg="bpass6")]
    pub fn method_selfmutness(&mut self) { }
}



// Add Method To Impl ----------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
impl Foo {
    pub fn add_method_to_impl1(&self) { }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner,associated_item_def_ids")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="owner,associated_item_def_ids")]
#[rustc_clean(cfg="bpass6")]
impl Foo {
    #[rustc_clean(cfg="bpass2", except="owner")]
    #[rustc_clean(cfg="bpass3")]
    #[rustc_clean(cfg="bpass5", except="owner")]
    #[rustc_clean(cfg="bpass6")]
    pub fn add_method_to_impl1(&self) { }

    #[rustc_clean(cfg="bpass3")]
    #[rustc_clean(cfg="bpass6")]
    pub fn add_method_to_impl2(&self) { }
}



// Add Method Parameter --------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
impl Foo {
    //------------------------------------------------------------------------------------
    //--------------------------
    //------------------------------------------------------------------------------------
    //--------------------------
    pub fn add_method_parameter(&self        ) { }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(except="owner", cfg="bpass5")]
#[rustc_clean(cfg="bpass6")]
impl Foo {
    #[rustc_clean(cfg="bpass2", except="owner,fn_sig,typeck_root,optimized_mir")]
    #[rustc_clean(cfg="bpass3")]
    #[rustc_clean(cfg="bpass5", except="owner,fn_sig,typeck_root,optimized_mir")]
    #[rustc_clean(cfg="bpass6")]
    pub fn add_method_parameter(&self, _: i32) { }
}



// Change Method Parameter Name ------------------------------------------------
#[cfg(any(bpass1,bpass4))]
impl Foo {
    //----------------------------------------------------------------------
    //--------------------------
    //----------------------------------------------------------------------
    //--------------------------
    pub fn change_method_parameter_name(&self, a: i64) { }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="owner")]
#[rustc_clean(cfg="bpass6")]
impl Foo {
    #[rustc_clean(cfg="bpass2", except="owner,optimized_mir")]
    #[rustc_clean(cfg="bpass3")]
    #[rustc_clean(cfg="bpass5", except="owner,optimized_mir")]
    #[rustc_clean(cfg="bpass6")]
    pub fn change_method_parameter_name(&self, b: i64) { }
}



// Change Method Return Type ---------------------------------------------------
#[cfg(any(bpass1,bpass4))]
impl Foo {
    //------------------------------------------------------------------------------------
    //--------------------------
    //------------------------------------------------------------------------------------
    //--------------------------
    pub fn change_method_return_type(&self) -> u16 { 0 }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(except="owner", cfg="bpass5")]
#[rustc_clean(cfg="bpass6")]
impl Foo {
    #[rustc_clean(cfg="bpass2", except="owner,fn_sig,optimized_mir,typeck_root")]
    #[rustc_clean(cfg="bpass3")]
    #[rustc_clean(cfg="bpass5", except="owner,fn_sig,optimized_mir,typeck_root")]
    #[rustc_clean(cfg="bpass6")]
    pub fn change_method_return_type(&self) -> u32 { 0 }
}



// Make Method #[inline] -------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
impl Foo {
    //--------------------------
    //--------------------------
    //--------------------------
    //--------------------------
    //-------
    pub fn make_method_inline(&self) -> u8 { 0 }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="owner")]
#[rustc_clean(cfg="bpass6")]
impl Foo {
    #[rustc_clean(cfg="bpass2", except="owner")]
    #[rustc_clean(cfg="bpass3")]
    #[rustc_clean(cfg="bpass5", except="owner")]
    #[rustc_clean(cfg="bpass6")]
    #[inline]
    pub fn make_method_inline(&self) -> u8 { 0 }
}



//  Change order of parameters -------------------------------------------------
#[cfg(any(bpass1,bpass4))]
impl Foo {
    //----------------------------------------------------------------------
    //--------------------------
    //----------------------------------------------------------------------
    //--------------------------
    pub fn change_method_parameter_order(&self, a: i64, b: i64) { }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="owner")]
#[rustc_clean(cfg="bpass6")]
impl Foo {
    #[rustc_clean(cfg="bpass2", except="owner,optimized_mir")]
    #[rustc_clean(cfg="bpass3")]
    #[rustc_clean(cfg="bpass5", except="owner,optimized_mir")]
    #[rustc_clean(cfg="bpass6")]
    pub fn change_method_parameter_order(&self, b: i64, a: i64) { }
}



// Make method unsafe ----------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
impl Foo {
    //----------------------------------------------------------------------
    //--------------------------
    //----------------------------------------------------------------------
    //--------------------------
    pub        fn make_method_unsafe(&self) { }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(except="owner", cfg="bpass5")]
#[rustc_clean(cfg="bpass6")]
impl Foo {
    #[rustc_clean(cfg="bpass2", except="owner,fn_sig,typeck_root")]
    #[rustc_clean(cfg="bpass3")]
    #[rustc_clean(cfg="bpass5", except="owner,fn_sig,typeck_root")]
    #[rustc_clean(cfg="bpass6")]
    pub unsafe fn make_method_unsafe(&self) { }
}



// Make method extern ----------------------------------------------------------
#[cfg(any(bpass1,bpass4))]
impl Foo {
    //----------------------------------------------------------------------
    //--------------------------
    //----------------------------------------------------------------------
    //--------------------------
    pub            fn make_method_extern(&self) { }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(except="owner", cfg="bpass5")]
#[rustc_clean(cfg="bpass6")]
impl Foo {
    #[rustc_clean(cfg="bpass2", except="owner,fn_sig,typeck_root")]
    #[rustc_clean(cfg="bpass3")]
    #[rustc_clean(cfg="bpass5", except="owner,fn_sig,typeck_root")]
    #[rustc_clean(cfg="bpass6")]
    pub extern "C" fn make_method_extern(&self) { }
}



// Change method calling convention --------------------------------------------
#[cfg(any(bpass1,bpass4))]
impl Foo {
    //----------------------------------------------------------------------
    //--------------------------
    //----------------------------------------------------------------------
    //--------------------------
    pub extern "C"      fn change_method_calling_convention(&self) { }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(except="owner", cfg="bpass5")]
#[rustc_clean(cfg="bpass6")]
impl Foo {
    #[rustc_clean(cfg="bpass2", except="owner,fn_sig,typeck_root")]
    #[rustc_clean(cfg="bpass3")]
    #[rustc_clean(cfg="bpass5", except="owner,fn_sig,typeck_root")]
    #[rustc_clean(cfg="bpass6")]
    pub extern "system" fn change_method_calling_convention(&self) { }
}



// Add Lifetime Parameter to Method --------------------------------------------
#[cfg(any(bpass1,bpass4))]
impl Foo {
    // -----------------------------------------------------
    // ---------------------------------------------------------
    // ----------------------------------------------------------
    // -------------------------------------------------------
    // -------------------------------------------------------
    // --------------------------------------------------------
    // ----------------------------------------------------------
    // -----------------------------------------------------------
    // ----------------------------------------------------------
    // --------------------------------------------------------------
    // -------------------------
    // --------------------------------------------------------------------------
    // -------------------------
    pub fn add_lifetime_parameter_to_method    (&self) { }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(except="owner", cfg="bpass5")]
#[rustc_clean(cfg="bpass6")]
impl Foo {
    // Warning: Note that `typeck_root` are coming up clean here.
    // The addition or removal of lifetime parameters that don't
    // appear in the arguments or fn body in any way does not, in
    // fact, affect the `typeck_root` in any semantic way (at least
    // as of this writing). **However,** altering the order of
    // lowering **can** cause it appear to affect the `typeck_root`:
    // if we lower generics before the body, then the `HirId` for
    // things in the body will be affected. So if you start to see
    // `typeck_root` appear dirty, that might be the cause. -nmatsakis
    #[rustc_clean(cfg="bpass2", except="owner,fn_sig")]
    #[rustc_clean(cfg="bpass3")]
    #[rustc_clean(cfg="bpass5", except="owner,fn_sig,generics_of")]
    #[rustc_clean(cfg="bpass6")]
    pub fn add_lifetime_parameter_to_method<'a>(&self) { }
}



// Add Type Parameter To Method ------------------------------------------------
#[cfg(any(bpass1,bpass4))]
impl Foo {
    // -----------------------------------------------------
    // ---------------------------------------------------------------
    // -------------------------------------------------------------
    // -----------------------------------------------------
    // -------------------------------------------------------------
    // ---------------------------------------------------
    // ------------------------------------------------------------
    // ------------------------------------------------------
    // -------------------------------------------------
    // -----------
    // --------------
    // ----------------------------------------------------------------
    //
    // -------------------------
    // -----------
    // --------------
    // ----------------------------------------------------------------
    //
    // -------------------------
    pub fn add_type_parameter_to_method   (&self) { }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(except="owner", cfg="bpass5")]
#[rustc_clean(cfg="bpass6")]
impl Foo {
    // Warning: Note that `typeck_root` are coming up clean here.
    // The addition or removal of type parameters that don't appear in
    // the arguments or fn body in any way does not, in fact, affect
    // the `typeck_root` in any semantic way (at least as of this
    // writing). **However,** altering the order of lowering **can**
    // cause it appear to affect the `typeck_root`: if we lower
    // generics before the body, then the `HirId` for things in the
    // body will be affected. So if you start to see `typeck_root`
    // appear dirty, that might be the cause. -nmatsakis
    #[rustc_clean(
        cfg="bpass2",
        except="owner,generics_of,predicates_of,type_of",
    )]
    #[rustc_clean(cfg="bpass3")]
    #[rustc_clean(
        cfg="bpass5",
        except="owner,generics_of,predicates_of,type_of",
    )]
    #[rustc_clean(cfg="bpass6")]
    pub fn add_type_parameter_to_method<T>(&self) { }
}



// Add Lifetime Bound to Lifetime Parameter of Method --------------------------
#[cfg(any(bpass1,bpass4))]
impl Foo {
    //------------
    //---------------
    //-----------------------------------------------------------------------
    //
    //--------------------------
    //------------
    //---------------
    //-----------------------------------------------------------------------
    //
    //--------------------------
    pub fn add_lifetime_bound_to_lifetime_param_of_method<'a, 'b    >(&self) { }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="owner")]
#[rustc_clean(cfg="bpass6")]
impl Foo {
    #[rustc_clean(
        cfg="bpass2",
        except="owner,generics_of,predicates_of,type_of,fn_sig"
    )]
    #[rustc_clean(cfg="bpass3")]
    #[rustc_clean(
        cfg="bpass5",
        except="owner,generics_of,predicates_of,type_of,fn_sig"
    )]
    #[rustc_clean(cfg="bpass6")]
    pub fn add_lifetime_bound_to_lifetime_param_of_method<'a, 'b: 'a>(&self) { }
}



// Add Lifetime Bound to Type Parameter of Method ------------------------------
#[cfg(any(bpass1,bpass4))]
impl Foo {
    // -----------------------------------------------------
    // ----------------------------------------------------------
    // -------------------------------------------------------------
    // -------------------------------------------------
    // -------------------------------------------------------------
    // ---------------------------------------------------
    // ------------------------------------------------------------
    // ------------------------------------------------------
    // -------------------------------------------------
    // -----------
    // --------------
    // ----------------------------------------------------------------------
    //
    // -------------------------
    // -----------
    // --------------
    // ----------------------------------------------------------------------
    //
    // -------------------------
    pub fn add_lifetime_bound_to_type_param_of_method<'a, T    >(&self) { }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(except="owner", cfg="bpass5")]
#[rustc_clean(cfg="bpass6")]
impl Foo {
    // Warning: Note that `typeck_root` are coming up clean here.
    // The addition or removal of bounds that don't appear in the
    // arguments or fn body in any way does not, in fact, affect the
    // `typeck_root` in any semantic way (at least as of this
    // writing). **However,** altering the order of lowering **can**
    // cause it appear to affect the `typeck_root`: if we lower
    // generics before the body, then the `HirId` for things in the
    // body will be affected. So if you start to see `typeck_root`
    // appear dirty, that might be the cause. -nmatsakis
    #[rustc_clean(
        cfg="bpass2",
        except="owner,generics_of,predicates_of,type_of,fn_sig"
    )]
    #[rustc_clean(cfg="bpass3")]
    #[rustc_clean(
        cfg="bpass5",
        except="owner,generics_of,predicates_of,type_of,fn_sig"
    )]
    #[rustc_clean(cfg="bpass6")]
    pub fn add_lifetime_bound_to_type_param_of_method<'a, T: 'a>(&self) { }
}



// Add Trait Bound to Type Parameter of Method ------------------------------
#[cfg(any(bpass1,bpass4))]
impl Foo {
    // -----------------------------------------------------
    // ----------------------------------------------------------
    // -------------------------------------------------------------
    // -------------------------------------------------
    // -------------------------------------------------------------
    // ---------------------------------------------------
    // ------------------------------------------------------------
    // ------------------------------------------------------
    // -------------------------------------------------
    // ---------------------------------------------------------------------
    // -------------------------
    // ---------------------------------------------------------------------
    // -------------------------
    pub fn add_trait_bound_to_type_param_of_method<T       >(&self) { }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(except="owner", cfg="bpass5")]
#[rustc_clean(cfg="bpass6")]
impl Foo {
    // Warning: Note that `typeck_root` are coming up clean here.
    // The addition or removal of bounds that don't appear in the
    // arguments or fn body in any way does not, in fact, affect the
    // `typeck_root` in any semantic way (at least as of this
    // writing). **However,** altering the order of lowering **can**
    // cause it appear to affect the `typeck_root`: if we lower
    // generics before the body, then the `HirId` for things in the
    // body will be affected. So if you start to see `typeck_root`
    // appear dirty, that might be the cause. -nmatsakis
    #[rustc_clean(cfg="bpass2", except="owner,predicates_of")]
    #[rustc_clean(cfg="bpass3")]
    #[rustc_clean(cfg="bpass5", except="owner,predicates_of")]
    #[rustc_clean(cfg="bpass6")]
    pub fn add_trait_bound_to_type_param_of_method<T: Clone>(&self) { }
}



// Add #[no_mangle] to Method --------------------------------------------------
#[cfg(any(bpass1,bpass4))]
impl Foo {
    //--------------------------
    //--------------------------
    //--------------------------
    //--------------------------
    //------------------
    pub fn add_no_mangle_to_method(&self) { }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="owner")]
#[rustc_clean(cfg="bpass6")]
impl Foo {
    #[rustc_clean(cfg="bpass2", except="owner")]
    #[rustc_clean(cfg="bpass3")]
    #[rustc_clean(cfg="bpass5", except="owner")]
    #[rustc_clean(cfg="bpass6")]
    #[unsafe(no_mangle)]
    pub fn add_no_mangle_to_method(&self) { }
}



struct Bar<T>(T);

// Add Type Parameter To Impl --------------------------------------------------
#[cfg(any(bpass1,bpass4))]
impl Bar<u32> {
    pub fn add_type_parameter_to_impl(&self) { }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner,generics_of")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="owner,generics_of")]
#[rustc_clean(cfg="bpass6")]
impl<T> Bar<T> {
    #[rustc_clean(
        cfg="bpass2",
        except="generics_of,fn_sig,typeck_root,type_of,optimized_mir,owner"
    )]
    #[rustc_clean(cfg="bpass3")]
    #[rustc_clean(
        cfg="bpass5",
        except="generics_of,fn_sig,typeck_root,type_of,optimized_mir,owner"
    )]
    #[rustc_clean(cfg="bpass6")]
    pub fn add_type_parameter_to_impl(&self) { }
}



// Change Self Type of Impl ----------------------------------------------------
#[cfg(any(bpass1,bpass4))]
impl Bar<u32> {
    pub fn change_impl_self_type(&self) { }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="owner")]
#[rustc_clean(cfg="bpass6")]
impl Bar<u64> {
    #[rustc_clean(cfg="bpass2", except="fn_sig,optimized_mir,typeck_root,owner")]
    #[rustc_clean(cfg="bpass3")]
    #[rustc_clean(cfg="bpass5", except="fn_sig,optimized_mir,typeck_root,owner")]
    #[rustc_clean(cfg="bpass6")]
    pub fn change_impl_self_type(&self) { }
}



// Add Lifetime Bound to Impl --------------------------------------------------
#[cfg(any(bpass1,bpass4))]
impl<T> Bar<T> {
    pub fn add_lifetime_bound_to_impl_parameter(&self) { }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="owner")]
#[rustc_clean(cfg="bpass6")]
impl<T: 'static> Bar<T> {
    #[rustc_clean(cfg="bpass2", except="owner")]
    #[rustc_clean(cfg="bpass3")]
    #[rustc_clean(cfg="bpass5", except="owner")]
    #[rustc_clean(cfg="bpass6")]
    pub fn add_lifetime_bound_to_impl_parameter(&self) { }
}



// Add Trait Bound to Impl Parameter -------------------------------------------
#[cfg(any(bpass1,bpass4))]
impl<T> Bar<T> {
    pub fn add_trait_bound_to_impl_parameter(&self) { }
}

#[cfg(not(any(bpass1,bpass4)))]
#[rustc_clean(cfg="bpass2", except="owner")]
#[rustc_clean(cfg="bpass3")]
#[rustc_clean(cfg="bpass5", except="owner")]
#[rustc_clean(cfg="bpass6")]
impl<T: Clone> Bar<T> {
    #[rustc_clean(cfg="bpass2", except="owner")]
    #[rustc_clean(cfg="bpass3")]
    #[rustc_clean(cfg="bpass5", except="owner")]
    #[rustc_clean(cfg="bpass6")]
    pub fn add_trait_bound_to_impl_parameter(&self) { }
}


// Force instantiation of some fns so we can check their hash.
pub fn instantiation_root() {
    Foo::method_privacy();

    #[cfg(any(bpass1,bpass4))]
    {
        Bar(0u32).change_impl_self_type();
    }

    #[cfg(not(any(bpass1,bpass4)))]
    {
        Bar(0u64).change_impl_self_type();
    }
}
