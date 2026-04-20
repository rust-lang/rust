// This test case tests the incremental compilation hash (ICH) implementation
// for trait definitions.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

// We also test the ICH for trait definitions exported in metadata. Same as
// above, we want to make sure that the change between rev1 and rev2 also
// results in a change of the ICH for the trait's metadata, and that it stays
// the same between rev2 and rev3.

//@ build-pass (FIXME(62277): could be check-pass?)
//@ revisions: bfail1 bfail2 bfail3 bfail4 bfail5 bfail6
//@ compile-flags: -Z query-dep-graph -O
//@ [bfail1]compile-flags: -Zincremental-ignore-spans
//@ [bfail2]compile-flags: -Zincremental-ignore-spans
//@ [bfail3]compile-flags: -Zincremental-ignore-spans
//@ ignore-backends: gcc

#![allow(warnings)]
#![feature(rustc_attrs)]
#![crate_type="rlib"]
#![feature(associated_type_defaults)]


// Change trait visibility
#[cfg(any(bfail1,bfail4))]
trait TraitVisibility { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5", except="opt_hir_owner_nodes,predicates_of")]
#[rustc_clean(cfg="bfail6")]
pub trait TraitVisibility { }



// Change trait unsafety
#[cfg(any(bfail1,bfail4))]
trait TraitUnsafety { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
unsafe trait TraitUnsafety { }



// Add method
#[cfg(any(bfail1,bfail4))]
trait TraitAddMethod {
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,associated_item_def_ids", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,associated_item_def_ids,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
pub trait TraitAddMethod {
    fn method();
}



// Change name of method
#[cfg(any(bfail1,bfail4))]
trait TraitChangeMethodName {
    fn method();
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,associated_item_def_ids", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,associated_item_def_ids,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitChangeMethodName {
    fn methodChanged();
}



// Add return type to method
#[cfg(any(bfail1,bfail4))]
trait TraitAddReturnType {
    //---------------------------------------------------------------
    //--------------------------
    //---------------------------------------------------------------
    //--------------------------
    fn method()       ;
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddReturnType {
    #[rustc_clean(except="opt_hir_owner_nodes,fn_sig", cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes,fn_sig", cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    fn method() -> u32;
}



// Change return type of method
#[cfg(any(bfail1,bfail4))]
trait TraitChangeReturnType {
    // --------------------------------------------------------------
    // -------------------------
    // --------------------------------------------------------------
    // -------------------------
    fn method() -> u32;
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitChangeReturnType {
    #[rustc_clean(except="opt_hir_owner_nodes,fn_sig", cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes,fn_sig", cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    fn method() -> u64;
}



// Add parameter to method
#[cfg(any(bfail1,bfail4))]
trait TraitAddParameterToMethod {
    // --------------------------------------------------------------
    // -------------------------
    // --------------------------------------------------------------
    // -------------------------
    fn method(      );
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddParameterToMethod {
    #[rustc_clean(except="opt_hir_owner_nodes,fn_sig", cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes,fn_sig", cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    fn method(a: u32);
}



// Change name of method parameter
#[cfg(any(bfail1,bfail4))]
trait TraitChangeMethodParameterName {
    //------------------------------------------------------
    //--------------------------------------------------------
    //--------------------------
    //--------------------------------------------------------
    //--------------------------
    fn method(a: u32);

    //----------------------------------------------------------------------
    //--------------------------
    //----------------------------------------------------------------------
    //--------------------------
    fn with_default(x: i32) {}
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitChangeMethodParameterName {
    // FIXME(#38501) This should preferably always be clean.
    #[rustc_clean(except="opt_hir_owner_nodes", cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes", cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    fn method(b: u32);

    #[rustc_clean(except="opt_hir_owner_nodes,optimized_mir", cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes,optimized_mir", cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    fn with_default(y: i32) {}
}



// Change type of method parameter (i32 => i64)
#[cfg(any(bfail1,bfail4))]
trait TraitChangeMethodParameterType {
    // --------------------------------------------------------------
    // -------------------------
    // --------------------------------------------------------------
    // -------------------------
    fn method(a: i32);
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitChangeMethodParameterType {
    #[rustc_clean(except="opt_hir_owner_nodes,fn_sig", cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes,fn_sig", cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    fn method(a: i64);
}



// Change type of method parameter (&i32 => &mut i32)
#[cfg(any(bfail1,bfail4))]
trait TraitChangeMethodParameterTypeRef {
    // --------------------------------------------------------------
    // -------------------------
    // --------------------------------------------------------------
    // -------------------------
    fn method(a: &    i32);
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitChangeMethodParameterTypeRef {
    #[rustc_clean(except="opt_hir_owner_nodes,fn_sig", cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes,fn_sig", cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    fn method(a: &mut i32);
}



// Change order of method parameters
#[cfg(any(bfail1,bfail4))]
trait TraitChangeMethodParametersOrder {
    // --------------------------------------------------------------
    // -------------------------
    // --------------------------------------------------------------
    // -------------------------
    fn method(a: i32, b: i64);
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitChangeMethodParametersOrder {
    #[rustc_clean(except="opt_hir_owner_nodes,fn_sig", cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes,fn_sig", cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    fn method(b: i64, a: i32);
}



// Add default implementation to method
#[cfg(any(bfail1,bfail4))]
trait TraitAddMethodAutoImplementation {
    // -------------------------------------------------------
    // -------------------------
    // -------------------------------------------------------
    // -------------------------
    fn method()  ;
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddMethodAutoImplementation {
    #[rustc_clean(except="opt_hir_owner_nodes", cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes", cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    fn method() {}
}



// Change order of methods
#[cfg(any(bfail1,bfail4))]
trait TraitChangeOrderOfMethods {
    fn method0();
    fn method1();
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,associated_item_def_ids", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,associated_item_def_ids", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitChangeOrderOfMethods {
    fn method1();
    fn method0();
}



// Change mode of self parameter
#[cfg(any(bfail1,bfail4))]
trait TraitChangeModeSelfRefToMut {
    // --------------------------------------------------------------
    // -------------------------
    // --------------------------------------------------------------
    // -------------------------
    fn method(&    self);
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitChangeModeSelfRefToMut {
    #[rustc_clean(except="opt_hir_owner_nodes,fn_sig", cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes,fn_sig", cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    fn method(&mut self);
}



#[cfg(any(bfail1,bfail4))]
trait TraitChangeModeSelfOwnToMut: Sized {
    // ----------------------------------------------------------------------------
    // -------------------------
    // ----------------------------------------------------------------------------
    // -------------------------
    fn method(    self) {}
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitChangeModeSelfOwnToMut: Sized {
    #[rustc_clean(except="opt_hir_owner_nodes,typeck_root,optimized_mir", cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes,typeck_root,optimized_mir", cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    fn method(mut self) {}
}



#[cfg(any(bfail1,bfail4))]
trait TraitChangeModeSelfOwnToRef {
    // --------------------------------------------------------------------------
    // -------------------------
    // --------------------------------------------------------------------------
    // -------------------------
    fn method( self);
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitChangeModeSelfOwnToRef {
    #[rustc_clean(except="opt_hir_owner_nodes,fn_sig,generics_of", cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes,fn_sig,generics_of", cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    fn method(&self);
}



// Add unsafe modifier to method
#[cfg(any(bfail1,bfail4))]
trait TraitAddUnsafeModifier {
    // --------------------------------------------------------------
    // -------------------------
    // --------------------------------------------------------------
    // -------------------------
    fn        method();
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddUnsafeModifier {
    #[rustc_clean(except="opt_hir_owner_nodes,fn_sig", cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes,fn_sig", cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    unsafe fn method();
}



// Add extern modifier to method
#[cfg(any(bfail1,bfail4))]
trait TraitAddExternModifier {
    // --------------------------------------------------------------
    // -------------------------
    // --------------------------------------------------------------
    // -------------------------
    fn            method();
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddExternModifier {
    #[rustc_clean(except="opt_hir_owner_nodes,fn_sig", cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes,fn_sig", cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    extern "C" fn method();
}



// Change extern "C" to extern "stdcall"
#[cfg(any(bfail1,bfail4))]
trait TraitChangeExternCToExternSystem {
    // --------------------------------------------------------------
    // -------------------------
    // --------------------------------------------------------------
    // -------------------------
    extern "C"       fn method();
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitChangeExternCToRustIntrinsic {
    #[rustc_clean(except="opt_hir_owner_nodes,fn_sig", cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes,fn_sig", cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    extern "system" fn method();
}



// Add type parameter to method
#[cfg(any(bfail1,bfail4))]
trait TraitAddTypeParameterToMethod {
    // --------------------------------------------------------------------------
    // ---------------
    // -------------------------
    // --------------------------------------------------------------------------
    // ---------------
    // -------------------------
    fn method   ();
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddTypeParameterToMethod {
    #[rustc_clean(except="opt_hir_owner_nodes,generics_of,predicates_of,type_of",
        cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes,generics_of,predicates_of,type_of",
        cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    fn method<T>();
}



// Add lifetime parameter to method
#[cfg(any(bfail1,bfail4))]
trait TraitAddLifetimeParameterToMethod {
    // --------------------------------------------------------------------------
    // -------------------------
    // --------------------------------------------------------------------------
    // -------------------------
    fn method    ();
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddLifetimeParameterToMethod {
    #[rustc_clean(except="opt_hir_owner_nodes,fn_sig,generics_of", cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes,fn_sig,generics_of", cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    fn method<'a>();
}



// dummy trait for bound
trait ReferencedTrait0 { }
trait ReferencedTrait1 { }

// Add trait bound to method type parameter
#[cfg(any(bfail1,bfail4))]
trait TraitAddTraitBoundToMethodTypeParameter {
    // ---------------------------------------------------------------------
    // -------------------------
    // ---------------------------------------------------------------------
    // -------------------------
    fn method<T                  >();
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddTraitBoundToMethodTypeParameter {
    #[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    fn method<T: ReferencedTrait0>();
}



// Add builtin bound to method type parameter
#[cfg(any(bfail1,bfail4))]
trait TraitAddBuiltinBoundToMethodTypeParameter {
    // ---------------------------------------------------------------------
    // -------------------------
    // ---------------------------------------------------------------------
    // -------------------------
    fn method<T       >();
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddBuiltinBoundToMethodTypeParameter {
    #[rustc_clean(except="opt_hir_owner_nodes", cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    fn method<T: Sized>();
}



// Add lifetime bound to method lifetime parameter
#[cfg(any(bfail1,bfail4))]
trait TraitAddLifetimeBoundToMethodLifetimeParameter {
    // -----------
    // -----------------------------------------------------------------------
    // --------------
    //
    // -------------------------
    // -----------
    // -----------------------------------------------------------------------
    // --------------
    //
    // -------------------------
    fn method<'a, 'b    >(a: &'a u32, b: &'b u32);
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddLifetimeBoundToMethodLifetimeParameter {
    #[rustc_clean(
        except="opt_hir_owner_nodes,generics_of,predicates_of,fn_sig,type_of",
        cfg="bfail2",
    )]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(
        except="opt_hir_owner_nodes,generics_of,predicates_of,fn_sig,type_of",
        cfg="bfail5",
    )]
    #[rustc_clean(cfg="bfail6")]
    fn method<'a, 'b: 'a>(a: &'a u32, b: &'b u32);
}



// Add second trait bound to method type parameter
#[cfg(any(bfail1,bfail4))]
trait TraitAddSecondTraitBoundToMethodTypeParameter {
    // ---------------------------------------------------------------------
    // -------------------------
    // ---------------------------------------------------------------------
    // -------------------------
    fn method<T: ReferencedTrait0                   >();
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddSecondTraitBoundToMethodTypeParameter {
    #[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    fn method<T: ReferencedTrait0 + ReferencedTrait1>();
}



// Add second builtin bound to method type parameter
#[cfg(any(bfail1,bfail4))]
trait TraitAddSecondBuiltinBoundToMethodTypeParameter {
    // ---------------------------------------------------------------------
    // -------------------------
    // ---------------------------------------------------------------------
    // -------------------------
    fn method<T: Sized       >();
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddSecondBuiltinBoundToMethodTypeParameter {
    #[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    fn method<T: Sized + Sync>();
}



// Add second lifetime bound to method lifetime parameter
#[cfg(any(bfail1,bfail4))]
trait TraitAddSecondLifetimeBoundToMethodLifetimeParameter {
    // -----------
    // -----------------------------------------------------------------------
    // --------------
    //
    // -------------------------
    // -----------
    // -----------------------------------------------------------------------
    // --------------
    //
    // -------------------------
    fn method<'a, 'b, 'c: 'a     >(a: &'a u32, b: &'b u32, c: &'c u32);
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddSecondLifetimeBoundToMethodLifetimeParameter {
    #[rustc_clean(
        except="opt_hir_owner_nodes,generics_of,predicates_of,fn_sig,type_of",
        cfg="bfail2",
    )]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(
        except="opt_hir_owner_nodes,generics_of,predicates_of,fn_sig,type_of",
        cfg="bfail5",
    )]
    #[rustc_clean(cfg="bfail6")]
    fn method<'a, 'b, 'c: 'a + 'b>(a: &'a u32, b: &'b u32, c: &'c u32);
}



// Add associated type
#[cfg(any(bfail1,bfail4))]
trait TraitAddAssociatedType {
    //--------------------------
    //--------------------------
    // -------------

    //--------------------------
    //--------------------------
    //--------------------------
    //--------------------------
    fn method();
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,associated_item_def_ids", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,associated_item_def_ids", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddAssociatedType {
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(cfg="bfail6")]
    type Associated;

    #[rustc_clean(cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    fn method();
}



// Add trait bound to associated type
#[cfg(any(bfail1,bfail4))]
trait TraitAddTraitBoundToAssociatedType {
    // -------------------------------------------------------
    // -------------------------
    // -------------------------------------------------------
    // -------------------------
    type Associated                  ;

    fn method();
}


// Apparently the type bound contributes to the predicates of the trait, but
// does not change the associated item itself.
#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddTraitBoundToAssociatedType {
    #[rustc_clean(except="opt_hir_owner_nodes", cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes", cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    type Associated: ReferencedTrait0;

    fn method();
}



// Add lifetime bound to associated type
#[cfg(any(bfail1,bfail4))]
trait TraitAddLifetimeBoundToAssociatedType<'a> {
    // -------------------------------------------------------
    // -------------------------
    // -------------------------------------------------------
    // -------------------------
    type Associated    ;

    fn method();
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddLifetimeBoundToAssociatedType<'a> {
    #[rustc_clean(except="opt_hir_owner_nodes", cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes", cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    type Associated: 'a;

    fn method();
}



// Add default to associated type
#[cfg(any(bfail1,bfail4))]
trait TraitAddDefaultToAssociatedType {
    //--------------------------------------------------------
    //--------------------------
    //--------------------------------------------------------
    //--------------------------
    type Associated                 ;

    fn method();
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddDefaultToAssociatedType {
    #[rustc_clean(except="opt_hir_owner_nodes", cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes", cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    type Associated = ReferenceType0;

    fn method();
}



// Add associated constant
#[cfg(any(bfail1,bfail4))]
trait TraitAddAssociatedConstant {
    fn method();
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,associated_item_def_ids", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,associated_item_def_ids,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddAssociatedConstant {
    const Value: u32;

    fn method();
}



// Add initializer to associated constant
#[cfg(any(bfail1,bfail4))]
trait TraitAddInitializerToAssociatedConstant {
    //--------------------------------------------------------
    //--------------------------
    //--------------------------------------------------------
    //--------------------------
    const Value: u32    ;

    //--------------------------
    //--------------------------
    //--------------------------
    //--------------------------
    fn method();
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddInitializerToAssociatedConstant {
    #[rustc_clean(except="opt_hir_owner_nodes", cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes", cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    const Value: u32 = 1;

    #[rustc_clean(cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    fn method();
}



// Change type of associated constant
#[cfg(any(bfail1,bfail4))]
trait TraitChangeTypeOfAssociatedConstant {
    // ---------------------------------------------------------------
    // -------------------------
    // ---------------------------------------------------------------
    // -------------------------
    const Value: u32;

    // -------------------------
    // -------------------------
    // -------------------------
    // -------------------------
    fn method();
}

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitChangeTypeOfAssociatedConstant {
    #[rustc_clean(except="opt_hir_owner_nodes,type_of", cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes,type_of", cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    const Value: f64;

    #[rustc_clean(cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    fn method();
}



// Add super trait
#[cfg(any(bfail1,bfail4))]
trait TraitAddSuperTrait { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddSuperTrait : ReferencedTrait0 { }



// Add builtin bound (Send or Copy)
#[cfg(any(bfail1,bfail4))]
trait TraitAddBuiltiBound { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddBuiltiBound : Send { }



// Add 'static lifetime bound to trait
#[cfg(any(bfail1,bfail4))]
trait TraitAddStaticLifetimeBound { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddStaticLifetimeBound : 'static { }



// Add super trait as second bound
#[cfg(any(bfail1,bfail4))]
trait TraitAddTraitAsSecondBound : ReferencedTrait0 { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddTraitAsSecondBound : ReferencedTrait0 + ReferencedTrait1 { }

#[cfg(any(bfail1,bfail4))]
trait TraitAddTraitAsSecondBoundFromBuiltin : Send { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddTraitAsSecondBoundFromBuiltin : Send + ReferencedTrait0 { }



// Add builtin bound as second bound
#[cfg(any(bfail1,bfail4))]
trait TraitAddBuiltinBoundAsSecondBound : ReferencedTrait0 { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddBuiltinBoundAsSecondBound : ReferencedTrait0 + Send { }

#[cfg(any(bfail1,bfail4))]
trait TraitAddBuiltinBoundAsSecondBoundFromBuiltin : Send { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddBuiltinBoundAsSecondBoundFromBuiltin: Send + Copy { }



// Add 'static bounds as second bound
#[cfg(any(bfail1,bfail4))]
trait TraitAddStaticBoundAsSecondBound : ReferencedTrait0 { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddStaticBoundAsSecondBound : ReferencedTrait0 + 'static { }

#[cfg(any(bfail1,bfail4))]
trait TraitAddStaticBoundAsSecondBoundFromBuiltin : Send { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddStaticBoundAsSecondBoundFromBuiltin : Send + 'static { }



// Add type parameter to trait
#[cfg(any(bfail1,bfail4))]
trait TraitAddTypeParameterToTrait { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,generics_of,predicates_of", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,generics_of,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddTypeParameterToTrait<T> { }



// Add lifetime parameter to trait
#[cfg(any(bfail1,bfail4))]
trait TraitAddLifetimeParameterToTrait { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,generics_of,predicates_of", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,generics_of,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddLifetimeParameterToTrait<'a> { }



// Add trait bound to type parameter of trait
#[cfg(any(bfail1,bfail4))]
trait TraitAddTraitBoundToTypeParameterOfTrait<T> { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddTraitBoundToTypeParameterOfTrait<T: ReferencedTrait0> { }



// Add lifetime bound to type parameter of trait
#[cfg(any(bfail1,bfail4))]
trait TraitAddLifetimeBoundToTypeParameterOfTrait<'a, T> { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddLifetimeBoundToTypeParameterOfTrait<'a, T: 'a> { }



// Add lifetime bound to lifetime parameter of trait
#[cfg(any(bfail1,bfail4))]
trait TraitAddLifetimeBoundToLifetimeParameterOfTrait<'a, 'b> { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddLifetimeBoundToLifetimeParameterOfTrait<'a: 'b, 'b> { }



// Add builtin bound to type parameter of trait
#[cfg(any(bfail1,bfail4))]
trait TraitAddBuiltinBoundToTypeParameterOfTrait<T> { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddBuiltinBoundToTypeParameterOfTrait<T: Send> { }



// Add second type parameter to trait
#[cfg(any(bfail1,bfail4))]
trait TraitAddSecondTypeParameterToTrait<T> { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,generics_of,predicates_of", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,generics_of,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddSecondTypeParameterToTrait<T, S> { }



// Add second lifetime parameter to trait
#[cfg(any(bfail1,bfail4))]
trait TraitAddSecondLifetimeParameterToTrait<'a> { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,generics_of,predicates_of", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,generics_of,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddSecondLifetimeParameterToTrait<'a, 'b> { }



// Add second trait bound to type parameter of trait
#[cfg(any(bfail1,bfail4))]
trait TraitAddSecondTraitBoundToTypeParameterOfTrait<T: ReferencedTrait0> { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddSecondTraitBoundToTypeParameterOfTrait<T: ReferencedTrait0 + ReferencedTrait1> { }



// Add second lifetime bound to type parameter of trait
#[cfg(any(bfail1,bfail4))]
trait TraitAddSecondLifetimeBoundToTypeParameterOfTrait<'a, 'b, T: 'a> { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddSecondLifetimeBoundToTypeParameterOfTrait<'a, 'b, T: 'a + 'b> { }



// Add second lifetime bound to lifetime parameter of trait
#[cfg(any(bfail1,bfail4))]
trait TraitAddSecondLifetimeBoundToLifetimeParameterOfTrait<'a: 'b, 'b, 'c> { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddSecondLifetimeBoundToLifetimeParameterOfTrait<'a: 'b + 'c, 'b, 'c> { }



// Add second builtin bound to type parameter of trait
#[cfg(any(bfail1,bfail4))]
trait TraitAddSecondBuiltinBoundToTypeParameterOfTrait<T: Send> { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddSecondBuiltinBoundToTypeParameterOfTrait<T: Send + Sync> { }



struct ReferenceType0 {}
struct ReferenceType1 {}



// Add trait bound to type parameter of trait in where clause
#[cfg(any(bfail1,bfail4))]
trait TraitAddTraitBoundToTypeParameterOfTraitWhere<T> { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddTraitBoundToTypeParameterOfTraitWhere<T> where T: ReferencedTrait0 { }



// Add lifetime bound to type parameter of trait in where clause
#[cfg(any(bfail1,bfail4))]
trait TraitAddLifetimeBoundToTypeParameterOfTraitWhere<'a, T> { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddLifetimeBoundToTypeParameterOfTraitWhere<'a, T> where T: 'a { }



// Add lifetime bound to lifetime parameter of trait in where clause
#[cfg(any(bfail1,bfail4))]
trait TraitAddLifetimeBoundToLifetimeParameterOfTraitWhere<'a, 'b> { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddLifetimeBoundToLifetimeParameterOfTraitWhere<'a, 'b> where 'a: 'b { }



// Add builtin bound to type parameter of trait in where clause
#[cfg(any(bfail1,bfail4))]
trait TraitAddBuiltinBoundToTypeParameterOfTraitWhere<T> { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddBuiltinBoundToTypeParameterOfTraitWhere<T> where T: Send { }



// Add second trait bound to type parameter of trait in where clause
#[cfg(any(bfail1,bfail4))]
trait TraitAddSecondTraitBoundToTypeParameterOfTraitWhere<T> where T: ReferencedTrait0 { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddSecondTraitBoundToTypeParameterOfTraitWhere<T>
    where T: ReferencedTrait0 + ReferencedTrait1 { }



// Add second lifetime bound to type parameter of trait in where clause
#[cfg(any(bfail1,bfail4))]
trait TraitAddSecondLifetimeBoundToTypeParameterOfTraitWhere<'a, 'b, T> where T: 'a { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddSecondLifetimeBoundToTypeParameterOfTraitWhere<'a, 'b, T> where T: 'a + 'b { }



// Add second lifetime bound to lifetime parameter of trait in where clause
#[cfg(any(bfail1,bfail4))]
trait TraitAddSecondLifetimeBoundToLifetimeParameterOfTraitWhere<'a, 'b, 'c> where 'a: 'b { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddSecondLifetimeBoundToLifetimeParameterOfTraitWhere<'a, 'b, 'c> where 'a: 'b + 'c { }



// Add second builtin bound to type parameter of trait in where clause
#[cfg(any(bfail1,bfail4))]
trait TraitAddSecondBuiltinBoundToTypeParameterOfTraitWhere<T> where T: Send { }

#[cfg(not(any(bfail1,bfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
#[rustc_clean(cfg="bfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
#[rustc_clean(cfg="bfail6")]
trait TraitAddSecondBuiltinBoundToTypeParameterOfTraitWhere<T> where T: Send + Sync { }


// Change return type of method indirectly by modifying a use statement
mod change_return_type_of_method_indirectly_use {
    #[cfg(any(bfail1,bfail4))]
    use super::ReferenceType0 as ReturnType;
    #[cfg(not(any(bfail1,bfail4)))]
    use super::ReferenceType1 as ReturnType;

    #[rustc_clean(cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    trait TraitChangeReturnType {
        #[rustc_clean(except="opt_hir_owner_nodes,fn_sig", cfg="bfail2")]
        #[rustc_clean(cfg="bfail3")]
        #[rustc_clean(except="opt_hir_owner_nodes,fn_sig", cfg="bfail5")]
        #[rustc_clean(cfg="bfail6")]
        fn method() -> ReturnType;
    }
}



// Change type of method parameter indirectly by modifying a use statement
mod change_method_parameter_type_indirectly_by_use {
    #[cfg(any(bfail1,bfail4))]
    use super::ReferenceType0 as ArgType;
    #[cfg(not(any(bfail1,bfail4)))]
    use super::ReferenceType1 as ArgType;

    #[rustc_clean(cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    trait TraitChangeArgType {
        #[rustc_clean(except="opt_hir_owner_nodes,fn_sig", cfg="bfail2")]
        #[rustc_clean(cfg="bfail3")]
        #[rustc_clean(except="opt_hir_owner_nodes,fn_sig", cfg="bfail5")]
        #[rustc_clean(cfg="bfail6")]
        fn method(a: ArgType);
    }
}



// Change trait bound of method type parameter indirectly by modifying a use statement
mod change_method_parameter_type_bound_indirectly_by_use {
    #[cfg(any(bfail1,bfail4))]
    use super::ReferencedTrait0 as Bound;
    #[cfg(not(any(bfail1,bfail4)))]
    use super::ReferencedTrait1 as Bound;

    #[rustc_clean(cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    trait TraitChangeBoundOfMethodTypeParameter {
        #[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
        #[rustc_clean(cfg="bfail3")]
        #[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
        #[rustc_clean(cfg="bfail6")]
        fn method<T: Bound>(a: T);
    }
}



// Change trait bound of method type parameter in where clause indirectly
// by modifying a use statement
mod change_method_parameter_type_bound_indirectly_by_use_where {
    #[cfg(any(bfail1,bfail4))]
    use super::ReferencedTrait0 as Bound;
    #[cfg(not(any(bfail1,bfail4)))]
    use super::ReferencedTrait1 as Bound;

    #[rustc_clean(cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    trait TraitChangeBoundOfMethodTypeParameterWhere {
        #[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
        #[rustc_clean(cfg="bfail3")]
        #[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
        #[rustc_clean(cfg="bfail6")]
        fn method<T>(a: T) where T: Bound;
    }
}



// Change trait bound of trait type parameter indirectly by modifying a use statement
mod change_method_type_parameter_bound_indirectly {
    #[cfg(any(bfail1,bfail4))]
    use super::ReferencedTrait0 as Bound;
    #[cfg(not(any(bfail1,bfail4)))]
    use super::ReferencedTrait1 as Bound;

    #[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    trait TraitChangeTraitBound<T: Bound> {
        fn method(a: T);
    }
}



// Change trait bound of trait type parameter in where clause indirectly
// by modifying a use statement
mod change_method_type_parameter_bound_indirectly_where {
    #[cfg(any(bfail1,bfail4))]
    use super::ReferencedTrait0 as Bound;
    #[cfg(not(any(bfail1,bfail4)))]
    use super::ReferencedTrait1 as Bound;

    #[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes,predicates_of", cfg="bfail5")]
    #[rustc_clean(cfg="bfail6")]
    trait TraitChangeTraitBoundWhere<T> where T: Bound {
        fn method(a: T);
    }
}
