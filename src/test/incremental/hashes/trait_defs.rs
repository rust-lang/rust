// This test case tests the incremental compilation hash (ICH) implementation
// for trait definitions.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

// We also test the ICH for trait definitions exported in metadata. Same as
// above, we want to make sure that the change between rev1 and rev2 also
// results in a change of the ICH for the trait's metadata, and that it stays
// the same between rev2 and rev3.

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
#![crate_type="rlib"]
#![feature(associated_type_defaults)]


// Change trait visibility
#[cfg(any(cfail1,cfail4))]
trait TraitVisibility { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub trait TraitVisibility { }



// Change trait unsafety
#[cfg(any(cfail1,cfail4))]
trait TraitUnsafety { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
unsafe trait TraitUnsafety { }



// Add method
#[cfg(any(cfail1,cfail4))]
trait TraitAddMethod {
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,associated_item_def_ids", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,associated_item_def_ids,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub trait TraitAddMethod {
    fn method();
}



// Change name of method
#[cfg(any(cfail1,cfail4))]
trait TraitChangeMethodName {
    fn method();
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,associated_item_def_ids", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,associated_item_def_ids", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitChangeMethodName {
    fn methodChanged();
}



// Add return type to method
#[cfg(any(cfail1,cfail4))]
trait TraitAddReturnType {
    //-----------------------------------------------------
    //--------------------------
    //-----------------------------------------------------
    //--------------------------
    fn method()       ;
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddReturnType {
    #[rustc_clean(except="hir_owner,fn_sig", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner,fn_sig", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    fn method() -> u32;
}



// Change return type of method
#[cfg(any(cfail1,cfail4))]
trait TraitChangeReturnType {
    // --------------------------------------------------------------------
    // -------------------------
    // --------------------------------------------------------------------
    // -------------------------
    fn method() -> u32;
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitChangeReturnType {
    #[rustc_clean(except="hir_owner,hir_owner_nodes,fn_sig", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner,hir_owner_nodes,fn_sig", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    fn method() -> u64;
}



// Add parameter to method
#[cfg(any(cfail1,cfail4))]
trait TraitAddParameterToMethod {
    // ----------------------------------------------------
    // -------------------------
    // ----------------------------------------------------
    // -------------------------
    fn method(      );
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddParameterToMethod {
    #[rustc_clean(except="hir_owner,fn_sig", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner,fn_sig", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    fn method(a: u32);
}



// Change name of method parameter
#[cfg(any(cfail1,cfail4))]
trait TraitChangeMethodParameterName {
    //------------------------------------------------------
    //----------------------------------------------
    //--------------------------
    //----------------------------------------------
    //--------------------------
    fn method(a: u32);

    //------------------------------------------------------------------
    //--------------------------
    //------------------------------------------------------------------
    //--------------------------
    fn with_default(x: i32) {}
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitChangeMethodParameterName {
    // FIXME(#38501) This should preferably always be clean.
    #[rustc_clean(except="hir_owner", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    fn method(b: u32);

    #[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    fn with_default(y: i32) {}
}



// Change type of method parameter (i32 => i64)
#[cfg(any(cfail1,cfail4))]
trait TraitChangeMethodParameterType {
    // ----------------------------------------------------
    // -------------------------
    // ----------------------------------------------------
    // -------------------------
    fn method(a: i32);
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitChangeMethodParameterType {
    #[rustc_clean(except="hir_owner,fn_sig", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner,fn_sig", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    fn method(a: i64);
}



// Change type of method parameter (&i32 => &mut i32)
#[cfg(any(cfail1,cfail4))]
trait TraitChangeMethodParameterTypeRef {
    // ----------------------------------------------------
    // -------------------------
    // ----------------------------------------------------
    // -------------------------
    fn method(a: &    i32);
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitChangeMethodParameterTypeRef {
    #[rustc_clean(except="hir_owner,fn_sig", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner,fn_sig", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    fn method(a: &mut i32);
}



// Change order of method parameters
#[cfg(any(cfail1,cfail4))]
trait TraitChangeMethodParametersOrder {
    // ----------------------------------------------------
    // -------------------------
    // ----------------------------------------------------
    // -------------------------
    fn method(a: i32, b: i64);
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitChangeMethodParametersOrder {
    #[rustc_clean(except="hir_owner,fn_sig", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner,fn_sig", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    fn method(b: i64, a: i32);
}



// Add default implementation to method
#[cfg(any(cfail1,cfail4))]
trait TraitAddMethodAutoImplementation {
    fn method();
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddMethodAutoImplementation {
    #[rustc_clean(except="hir_owner,associated_item", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner,associated_item", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    fn method() { }
}



// Change order of methods
#[cfg(any(cfail1,cfail4))]
trait TraitChangeOrderOfMethods {
    fn method0();
    fn method1();
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,associated_item_def_ids", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,associated_item_def_ids", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitChangeOrderOfMethods {
    fn method1();
    fn method0();
}



// Change mode of self parameter
#[cfg(any(cfail1,cfail4))]
trait TraitChangeModeSelfRefToMut {
    // ----------------------------------------------------
    // -------------------------
    // ----------------------------------------------------
    // -------------------------
    fn method(&    self);
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitChangeModeSelfRefToMut {
    #[rustc_clean(except="hir_owner,fn_sig", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner,fn_sig", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    fn method(&mut self);
}



#[cfg(any(cfail1,cfail4))]
trait TraitChangeModeSelfOwnToMut: Sized {
    // ----------------------------------------------------------------------------------
    // -------------------------
    // ----------------------------------------------------------------------------------
    // -------------------------
    fn method(    self) {}
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitChangeModeSelfOwnToMut: Sized {
    #[rustc_clean(except="hir_owner,hir_owner_nodes,typeck,optimized_mir", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner,hir_owner_nodes,typeck,optimized_mir", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    fn method(mut self) {}
}



#[cfg(any(cfail1,cfail4))]
trait TraitChangeModeSelfOwnToRef {
    // ----------------------------------------------------------------
    // -------------------------
    // ----------------------------------------------------------------
    // -------------------------
    fn method( self);
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitChangeModeSelfOwnToRef {
    #[rustc_clean(except="hir_owner,fn_sig,generics_of", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner,fn_sig,generics_of", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    fn method(&self);
}



// Add unsafe modifier to method
#[cfg(any(cfail1,cfail4))]
trait TraitAddUnsafeModifier {
    // ----------------------------------------------------
    // -------------------------
    // ----------------------------------------------------
    // -------------------------
    fn method()       ;
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddUnsafeModifier {
    #[rustc_clean(except="hir_owner,fn_sig", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner,fn_sig", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    unsafe fn method();
}



// Add extern modifier to method
#[cfg(any(cfail1,cfail4))]
trait TraitAddExternModifier {
    // ----------------------------------------------------
    // -------------------------
    // ----------------------------------------------------
    // -------------------------
    fn method()           ;
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddExternModifier {
    #[rustc_clean(except="hir_owner,fn_sig", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner,fn_sig", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    extern "C" fn method();
}



// Change extern "C" to extern "stdcall"
#[cfg(any(cfail1,cfail4))]
trait TraitChangeExternCToRustIntrinsic {
    // ----------------------------------------------------
    // -------------------------
    // ----------------------------------------------------
    // -------------------------
    extern "C"       fn method();
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitChangeExternCToRustIntrinsic {
    #[rustc_clean(except="hir_owner,fn_sig", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner,fn_sig", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    extern "stdcall" fn method();
}



// Add type parameter to method
#[cfg(any(cfail1,cfail4))]
trait TraitAddTypeParameterToMethod {
    // -------------------------------------------------------------------------------
    // -------------------------
    // -------------------------------------------------------------------------------
    // -------------------------
    fn method   ();
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddTypeParameterToMethod {
    #[rustc_clean(except="hir_owner,generics_of,predicates_of,type_of", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner,generics_of,predicates_of,type_of", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    fn method<T>();
}



// Add lifetime parameter to method
#[cfg(any(cfail1,cfail4))]
trait TraitAddLifetimeParameterToMethod {
    // ----------------------------------------------------------------
    // -------------------------
    // ----------------------------------------------------------------
    // -------------------------
    fn method    ();
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddLifetimeParameterToMethod {
    #[rustc_clean(except="hir_owner,fn_sig,generics_of", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner,fn_sig,generics_of", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    fn method<'a>();
}



// dummy trait for bound
trait ReferencedTrait0 { }
trait ReferencedTrait1 { }

// Add trait bound to method type parameter
#[cfg(any(cfail1,cfail4))]
trait TraitAddTraitBoundToMethodTypeParameter {
    // ---------------------------------------------------------------------------
    // -------------------------
    // ---------------------------------------------------------------------------
    // -------------------------
    fn method<T                  >();
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddTraitBoundToMethodTypeParameter {
    #[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    fn method<T: ReferencedTrait0>();
}



// Add builtin bound to method type parameter
#[cfg(any(cfail1,cfail4))]
trait TraitAddBuiltinBoundToMethodTypeParameter {
    // ---------------------------------------------------------------------------
    // -------------------------
    // ---------------------------------------------------------------------------
    // -------------------------
    fn method<T       >();
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddBuiltinBoundToMethodTypeParameter {
    #[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    fn method<T: Sized>();
}



// Add lifetime bound to method lifetime parameter
#[cfg(any(cfail1,cfail4))]
trait TraitAddLifetimeBoundToMethodLifetimeParameter {
    // -----------
    // -----------------------------------------------------------------------------
    // --------------
    //
    // -------------------------
    // -----------
    // -----------------------------------------------------------------------------
    // --------------
    //
    // -------------------------
    fn method<'a, 'b    >(a: &'a u32, b: &'b u32);
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddLifetimeBoundToMethodLifetimeParameter {
    #[rustc_clean(
        except="hir_owner,hir_owner_nodes,generics_of,predicates_of,fn_sig,type_of",
        cfg="cfail2",
    )]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(
        except="hir_owner,hir_owner_nodes,generics_of,predicates_of,fn_sig,type_of",
        cfg="cfail5",
    )]
    #[rustc_clean(cfg="cfail6")]
    fn method<'a, 'b: 'a>(a: &'a u32, b: &'b u32);
}



// Add second trait bound to method type parameter
#[cfg(any(cfail1,cfail4))]
trait TraitAddSecondTraitBoundToMethodTypeParameter {
    // ---------------------------------------------------------------------------
    // -------------------------
    // ---------------------------------------------------------------------------
    // -------------------------
    fn method<T: ReferencedTrait0                   >();
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddSecondTraitBoundToMethodTypeParameter {
    #[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    fn method<T: ReferencedTrait0 + ReferencedTrait1>();
}



// Add second builtin bound to method type parameter
#[cfg(any(cfail1,cfail4))]
trait TraitAddSecondBuiltinBoundToMethodTypeParameter {
    // ---------------------------------------------------------------------------
    // -------------------------
    // ---------------------------------------------------------------------------
    // -------------------------
    fn method<T: Sized       >();
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddSecondBuiltinBoundToMethodTypeParameter {
    #[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    fn method<T: Sized + Sync>();
}



// Add second lifetime bound to method lifetime parameter
#[cfg(any(cfail1,cfail4))]
trait TraitAddSecondLifetimeBoundToMethodLifetimeParameter {
    // -----------
    // -----------------------------------------------------------------------------
    // --------------
    //
    // -------------------------
    // -----------
    // -----------------------------------------------------------------------------
    // --------------
    //
    // -------------------------
    fn method<'a, 'b, 'c: 'a     >(a: &'a u32, b: &'b u32, c: &'c u32);
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddSecondLifetimeBoundToMethodLifetimeParameter {
    #[rustc_clean(
        except="hir_owner,hir_owner_nodes,generics_of,predicates_of,fn_sig,type_of",
        cfg="cfail2",
    )]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(
        except="hir_owner,hir_owner_nodes,generics_of,predicates_of,fn_sig,type_of",
        cfg="cfail5",
    )]
    #[rustc_clean(cfg="cfail6")]
    fn method<'a, 'b, 'c: 'a + 'b>(a: &'a u32, b: &'b u32, c: &'c u32);
}



// Add associated type
#[cfg(any(cfail1,cfail4))]
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

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,associated_item_def_ids", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,associated_item_def_ids", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddAssociatedType {
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(cfg="cfail6")]
    type Associated;

    #[rustc_clean(cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    fn method();
}



// Add trait bound to associated type
#[cfg(any(cfail1,cfail4))]
trait TraitAddTraitBoundToAssociatedType {
    // ---------------------------------------------
    // -------------------------
    // ---------------------------------------------
    // -------------------------
    type Associated                  ;

    fn method();
}


// Apparently the type bound contributes to the predicates of the trait, but
// does not change the associated item itself.
#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddTraitBoundToAssociatedType {
    #[rustc_clean(except="hir_owner", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    type Associated: ReferencedTrait0;

    fn method();
}



// Add lifetime bound to associated type
#[cfg(any(cfail1,cfail4))]
trait TraitAddLifetimeBoundToAssociatedType<'a> {
    // ---------------------------------------------
    // -------------------------
    // ---------------------------------------------
    // -------------------------
    type Associated    ;

    fn method();
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddLifetimeBoundToAssociatedType<'a> {
    #[rustc_clean(except="hir_owner", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    type Associated: 'a;

    fn method();
}



// Add default to associated type
#[cfg(any(cfail1,cfail4))]
trait TraitAddDefaultToAssociatedType {
    type Associated;

    fn method();
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddDefaultToAssociatedType {
    #[rustc_clean(except="hir_owner,associated_item", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner,associated_item", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    type Associated = ReferenceType0;

    fn method();
}



// Add associated constant
#[cfg(any(cfail1,cfail4))]
trait TraitAddAssociatedConstant {
    fn method();
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,associated_item_def_ids", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,associated_item_def_ids", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddAssociatedConstant {
    const Value: u32;

    fn method();
}



// Add initializer to associated constant
#[cfg(any(cfail1,cfail4))]
trait TraitAddInitializerToAssociatedConstant {
    const Value: u32;

    fn method();
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddInitializerToAssociatedConstant {
    #[rustc_clean(except="hir_owner,associated_item", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner,associated_item", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    const Value: u32 = 1;

    #[rustc_clean(cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    fn method();
}



// Change type of associated constant
#[cfg(any(cfail1,cfail4))]
trait TraitChangeTypeOfAssociatedConstant {
    // -----------------------------------------------------
    // -------------------------
    // -----------------------------------------------------
    // -------------------------
    const Value: u32;

    // -------------------------
    // -------------------------
    // -------------------------
    // -------------------------
    fn method();
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitChangeTypeOfAssociatedConstant {
    #[rustc_clean(except="hir_owner,type_of", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner,type_of", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    const Value: f64;

    #[rustc_clean(cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    fn method();
}



// Add super trait
#[cfg(any(cfail1,cfail4))]
trait TraitAddSuperTrait { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddSuperTrait : ReferencedTrait0 { }



// Add builtin bound (Send or Copy)
#[cfg(any(cfail1,cfail4))]
trait TraitAddBuiltiBound { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddBuiltiBound : Send { }



// Add 'static lifetime bound to trait
#[cfg(any(cfail1,cfail4))]
trait TraitAddStaticLifetimeBound { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddStaticLifetimeBound : 'static { }



// Add super trait as second bound
#[cfg(any(cfail1,cfail4))]
trait TraitAddTraitAsSecondBound : ReferencedTrait0 { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddTraitAsSecondBound : ReferencedTrait0 + ReferencedTrait1 { }

#[cfg(any(cfail1,cfail4))]
trait TraitAddTraitAsSecondBoundFromBuiltin : Send { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddTraitAsSecondBoundFromBuiltin : Send + ReferencedTrait0 { }



// Add builtin bound as second bound
#[cfg(any(cfail1,cfail4))]
trait TraitAddBuiltinBoundAsSecondBound : ReferencedTrait0 { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddBuiltinBoundAsSecondBound : ReferencedTrait0 + Send { }

#[cfg(any(cfail1,cfail4))]
trait TraitAddBuiltinBoundAsSecondBoundFromBuiltin : Send { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddBuiltinBoundAsSecondBoundFromBuiltin: Send + Copy { }



// Add 'static bounds as second bound
#[cfg(any(cfail1,cfail4))]
trait TraitAddStaticBoundAsSecondBound : ReferencedTrait0 { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddStaticBoundAsSecondBound : ReferencedTrait0 + 'static { }

#[cfg(any(cfail1,cfail4))]
trait TraitAddStaticBoundAsSecondBoundFromBuiltin : Send { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddStaticBoundAsSecondBoundFromBuiltin : Send + 'static { }



// Add type parameter to trait
#[cfg(any(cfail1,cfail4))]
trait TraitAddTypeParameterToTrait { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,generics_of,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,generics_of,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddTypeParameterToTrait<T> { }



// Add lifetime parameter to trait
#[cfg(any(cfail1,cfail4))]
trait TraitAddLifetimeParameterToTrait { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,generics_of,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,generics_of,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddLifetimeParameterToTrait<'a> { }



// Add trait bound to type parameter of trait
#[cfg(any(cfail1,cfail4))]
trait TraitAddTraitBoundToTypeParameterOfTrait<T> { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddTraitBoundToTypeParameterOfTrait<T: ReferencedTrait0> { }



// Add lifetime bound to type parameter of trait
#[cfg(any(cfail1,cfail4))]
trait TraitAddLifetimeBoundToTypeParameterOfTrait<'a, T> { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes,generics_of,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes,generics_of,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddLifetimeBoundToTypeParameterOfTrait<'a, T: 'a> { }



// Add lifetime bound to lifetime parameter of trait
#[cfg(any(cfail1,cfail4))]
trait TraitAddLifetimeBoundToLifetimeParameterOfTrait<'a, 'b> { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddLifetimeBoundToLifetimeParameterOfTrait<'a: 'b, 'b> { }



// Add builtin bound to type parameter of trait
#[cfg(any(cfail1,cfail4))]
trait TraitAddBuiltinBoundToTypeParameterOfTrait<T> { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddBuiltinBoundToTypeParameterOfTrait<T: Send> { }



// Add second type parameter to trait
#[cfg(any(cfail1,cfail4))]
trait TraitAddSecondTypeParameterToTrait<T> { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes,generics_of,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes,generics_of,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddSecondTypeParameterToTrait<T, S> { }



// Add second lifetime parameter to trait
#[cfg(any(cfail1,cfail4))]
trait TraitAddSecondLifetimeParameterToTrait<'a> { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes,generics_of,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes,generics_of,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddSecondLifetimeParameterToTrait<'a, 'b> { }



// Add second trait bound to type parameter of trait
#[cfg(any(cfail1,cfail4))]
trait TraitAddSecondTraitBoundToTypeParameterOfTrait<T: ReferencedTrait0> { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddSecondTraitBoundToTypeParameterOfTrait<T: ReferencedTrait0 + ReferencedTrait1> { }



// Add second lifetime bound to type parameter of trait
#[cfg(any(cfail1,cfail4))]
trait TraitAddSecondLifetimeBoundToTypeParameterOfTrait<'a, 'b, T: 'a> { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes,generics_of,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes,generics_of,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddSecondLifetimeBoundToTypeParameterOfTrait<'a, 'b, T: 'a + 'b> { }



// Add second lifetime bound to lifetime parameter of trait
#[cfg(any(cfail1,cfail4))]
trait TraitAddSecondLifetimeBoundToLifetimeParameterOfTrait<'a: 'b, 'b, 'c> { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddSecondLifetimeBoundToLifetimeParameterOfTrait<'a: 'b + 'c, 'b, 'c> { }



// Add second builtin bound to type parameter of trait
#[cfg(any(cfail1,cfail4))]
trait TraitAddSecondBuiltinBoundToTypeParameterOfTrait<T: Send> { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddSecondBuiltinBoundToTypeParameterOfTrait<T: Send + Sync> { }



struct ReferenceType0 {}
struct ReferenceType1 {}



// Add trait bound to type parameter of trait in where clause
#[cfg(any(cfail1,cfail4))]
trait TraitAddTraitBoundToTypeParameterOfTraitWhere<T> { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddTraitBoundToTypeParameterOfTraitWhere<T> where T: ReferencedTrait0 { }



// Add lifetime bound to type parameter of trait in where clause
#[cfg(any(cfail1,cfail4))]
trait TraitAddLifetimeBoundToTypeParameterOfTraitWhere<'a, T> { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes,generics_of,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes,generics_of,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddLifetimeBoundToTypeParameterOfTraitWhere<'a, T> where T: 'a { }



// Add lifetime bound to lifetime parameter of trait in where clause
#[cfg(any(cfail1,cfail4))]
trait TraitAddLifetimeBoundToLifetimeParameterOfTraitWhere<'a, 'b> { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddLifetimeBoundToLifetimeParameterOfTraitWhere<'a, 'b> where 'a: 'b { }



// Add builtin bound to type parameter of trait in where clause
#[cfg(any(cfail1,cfail4))]
trait TraitAddBuiltinBoundToTypeParameterOfTraitWhere<T> { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddBuiltinBoundToTypeParameterOfTraitWhere<T> where T: Send { }



// Add second trait bound to type parameter of trait in where clause
#[cfg(any(cfail1,cfail4))]
trait TraitAddSecondTraitBoundToTypeParameterOfTraitWhere<T> where T: ReferencedTrait0 { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddSecondTraitBoundToTypeParameterOfTraitWhere<T>
    where T: ReferencedTrait0 + ReferencedTrait1 { }



// Add second lifetime bound to type parameter of trait in where clause
#[cfg(any(cfail1,cfail4))]
trait TraitAddSecondLifetimeBoundToTypeParameterOfTraitWhere<'a, 'b, T> where T: 'a { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes,generics_of,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes,generics_of,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddSecondLifetimeBoundToTypeParameterOfTraitWhere<'a, 'b, T> where T: 'a + 'b { }



// Add second lifetime bound to lifetime parameter of trait in where clause
#[cfg(any(cfail1,cfail4))]
trait TraitAddSecondLifetimeBoundToLifetimeParameterOfTraitWhere<'a, 'b, 'c> where 'a: 'b { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddSecondLifetimeBoundToLifetimeParameterOfTraitWhere<'a, 'b, 'c> where 'a: 'b + 'c { }



// Add second builtin bound to type parameter of trait in where clause
#[cfg(any(cfail1,cfail4))]
trait TraitAddSecondBuiltinBoundToTypeParameterOfTraitWhere<T> where T: Send { }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
trait TraitAddSecondBuiltinBoundToTypeParameterOfTraitWhere<T> where T: Send + Sync { }


// Change return type of method indirectly by modifying a use statement
mod change_return_type_of_method_indirectly_use {
    #[cfg(any(cfail1,cfail4))]
    use super::ReferenceType0 as ReturnType;
    #[cfg(not(any(cfail1,cfail4)))]
    use super::ReferenceType1 as ReturnType;

    #[rustc_clean(cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    trait TraitChangeReturnType {
        #[rustc_clean(except="hir_owner,hir_owner_nodes,fn_sig", cfg="cfail2")]
        #[rustc_clean(cfg="cfail3")]
        #[rustc_clean(except="hir_owner,hir_owner_nodes,fn_sig", cfg="cfail5")]
        #[rustc_clean(cfg="cfail6")]
        fn method() -> ReturnType;
    }
}



// Change type of method parameter indirectly by modifying a use statement
mod change_method_parameter_type_indirectly_by_use {
    #[cfg(any(cfail1,cfail4))]
    use super::ReferenceType0 as ArgType;
    #[cfg(not(any(cfail1,cfail4)))]
    use super::ReferenceType1 as ArgType;

    #[rustc_clean(cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    trait TraitChangeArgType {
        #[rustc_clean(except="hir_owner,fn_sig", cfg="cfail2")]
        #[rustc_clean(cfg="cfail3")]
        fn method(a: ArgType);
    }
}



// Change trait bound of method type parameter indirectly by modifying a use statement
mod change_method_parameter_type_bound_indirectly_by_use {
    #[cfg(any(cfail1,cfail4))]
    use super::ReferencedTrait0 as Bound;
    #[cfg(not(any(cfail1,cfail4)))]
    use super::ReferencedTrait1 as Bound;

    #[rustc_clean(cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    trait TraitChangeBoundOfMethodTypeParameter {
        #[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail2")]
        #[rustc_clean(cfg="cfail3")]
        fn method<T: Bound>(a: T);
    }
}



// Change trait bound of method type parameter in where clause indirectly
// by modifying a use statement
mod change_method_parameter_type_bound_indirectly_by_use_where {
    #[cfg(any(cfail1,cfail4))]
    use super::ReferencedTrait0 as Bound;
    #[cfg(not(any(cfail1,cfail4)))]
    use super::ReferencedTrait1 as Bound;

    #[rustc_clean(cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    trait TraitChangeBoundOfMethodTypeParameterWhere {
        #[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail2")]
        #[rustc_clean(cfg="cfail3")]
        #[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail5")]
        #[rustc_clean(cfg="cfail6")]
        fn method<T>(a: T) where T: Bound;
    }
}



// Change trait bound of trait type parameter indirectly by modifying a use statement
mod change_method_type_parameter_bound_indirectly {
    #[cfg(any(cfail1,cfail4))]
    use super::ReferencedTrait0 as Bound;
    #[cfg(not(any(cfail1,cfail4)))]
    use super::ReferencedTrait1 as Bound;

    #[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    trait TraitChangeTraitBound<T: Bound> {
        fn method(a: T);
    }
}



// Change trait bound of trait type parameter in where clause indirectly
// by modifying a use statement
mod change_method_type_parameter_bound_indirectly_where {
    #[cfg(any(cfail1,cfail4))]
    use super::ReferencedTrait0 as Bound;
    #[cfg(not(any(cfail1,cfail4)))]
    use super::ReferencedTrait1 as Bound;

    #[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    trait TraitChangeTraitBoundWhere<T> where T: Bound {
        fn method(a: T);
    }
}
