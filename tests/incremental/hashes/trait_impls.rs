// This test case tests the incremental compilation hash (ICH) implementation
// for let expressions.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

//@ build-pass (FIXME(62277): could be check-pass?)
//@ revisions: cfail1 cfail2 cfail3 cfail4 cfail5 cfail6
//@ compile-flags: -Z query-dep-graph -O
//@ [cfail1]compile-flags: -Zincremental-ignore-spans
//@ [cfail2]compile-flags: -Zincremental-ignore-spans
//@ [cfail3]compile-flags: -Zincremental-ignore-spans

#![allow(warnings)]
#![feature(rustc_attrs)]
#![feature(specialization)]
#![crate_type="rlib"]

struct Foo;

// Change Method Name -----------------------------------------------------------

#[cfg(any(cfail1,cfail4))]
pub trait ChangeMethodNameTrait {
    fn method_name();
}

#[cfg(any(cfail1,cfail4))]
impl ChangeMethodNameTrait for Foo {
    fn method_name() { }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,associated_item_def_ids", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,associated_item_def_ids,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub trait ChangeMethodNameTrait {
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(cfg="cfail6")]
    fn method_name2();
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,associated_item_def_ids", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,associated_item_def_ids", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
impl ChangeMethodNameTrait for Foo {
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(cfg="cfail6")]
    fn method_name2() { }
}

// Change Method Body -----------------------------------------------------------
//
// This should affect the method itself, but not the impl.

pub trait ChangeMethodBodyTrait {
    fn method_name();
}

#[cfg(any(cfail1,cfail4))]
impl ChangeMethodBodyTrait for Foo {
    // --------------------------------------------------------------
    // -------------------------
    // --------------------------------------------------------------
    // -------------------------
    fn method_name() {
        //
    }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
impl ChangeMethodBodyTrait for Foo {
    #[rustc_clean(except="opt_hir_owner_nodes,typeck", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes,typeck", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    fn method_name() {
        ()
    }
}

// Change Method Body (inlined fn) ---------------------------------------------
//
// This should affect the method itself, but not the impl.

pub trait ChangeMethodBodyTraitInlined {
    fn method_name();
}

#[cfg(any(cfail1,cfail4))]
impl ChangeMethodBodyTraitInlined for Foo {
    // ----------------------------------------------------------------------------
    // -------------------------
    // ----------------------------------------------------------------------------
    // -------------------------
    #[inline]
    fn method_name() {
        // -----
    }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
impl ChangeMethodBodyTraitInlined for Foo {
    #[rustc_clean(except="opt_hir_owner_nodes,typeck,optimized_mir", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes,typeck,optimized_mir", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    #[inline]
    fn method_name() {
        panic!()
    }
}

// Change Method Selfness ------------------------------------------------------

#[cfg(any(cfail1,cfail4))]
pub trait ChangeMethodSelfnessTrait {
    fn method_name();
}

#[cfg(any(cfail1,cfail4))]
impl ChangeMethodSelfnessTrait for Foo {
    fn method_name() { }
}

#[cfg(not(any(cfail1,cfail4)))]
pub trait ChangeMethodSelfnessTrait {
    fn method_name(&self);
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="opt_hir_owner_nodes", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
impl ChangeMethodSelfnessTrait for Foo {
    #[rustc_clean(
        except="opt_hir_owner_nodes,associated_item,generics_of,fn_sig,typeck,optimized_mir",
        cfg="cfail2",
    )]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(
        except="opt_hir_owner_nodes,associated_item,generics_of,fn_sig,typeck,optimized_mir",
        cfg="cfail5",
    )]
    #[rustc_clean(cfg="cfail6")]
    fn method_name(&self) {
        ()
    }
}

// Change Method Selfness -----------------------------------------------------------

#[cfg(any(cfail1,cfail4))]
pub trait RemoveMethodSelfnessTrait {
    fn method_name(&self);
}

#[cfg(any(cfail1,cfail4))]
impl RemoveMethodSelfnessTrait for Foo {
    fn method_name(&self) { }
}

#[cfg(not(any(cfail1,cfail4)))]
pub trait RemoveMethodSelfnessTrait {
    fn method_name();
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="opt_hir_owner_nodes", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
impl RemoveMethodSelfnessTrait for Foo {
    #[rustc_clean(
        except="opt_hir_owner_nodes,associated_item,generics_of,fn_sig,typeck,optimized_mir",
        cfg="cfail2",
    )]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(
        except="opt_hir_owner_nodes,associated_item,generics_of,fn_sig,typeck,optimized_mir",
        cfg="cfail5",
    )]
    #[rustc_clean(cfg="cfail6")]
    fn method_name() {}
}

// Change Method Selfmutness -----------------------------------------------------------

#[cfg(any(cfail1,cfail4))]
pub trait ChangeMethodSelfmutnessTrait {
    fn method_name(&self);
}

#[cfg(any(cfail1,cfail4))]
impl ChangeMethodSelfmutnessTrait for Foo {
    // -----------------------------------------------------------------------------------
    // -------------------------
    // -----------------------------------------------------------------------------------
    // -------------------------
    fn method_name(&    self) {}
}

#[cfg(not(any(cfail1,cfail4)))]
pub trait ChangeMethodSelfmutnessTrait {
    fn method_name(&mut self);
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
impl ChangeMethodSelfmutnessTrait for Foo {
    #[rustc_clean(except="opt_hir_owner_nodes,fn_sig,typeck,optimized_mir", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes,fn_sig,typeck,optimized_mir", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    fn method_name(&mut self) {}
}

// Change item kind -----------------------------------------------------------

#[cfg(any(cfail1,cfail4))]
pub trait ChangeItemKindTrait {
    fn name();
}

#[cfg(any(cfail1,cfail4))]
impl ChangeItemKindTrait for Foo {
    fn name() { }
}

#[cfg(not(any(cfail1,cfail4)))]
pub trait ChangeItemKindTrait {
    type name;
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,associated_item_def_ids", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,associated_item_def_ids", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
impl ChangeItemKindTrait for Foo {
    type name = ();
}

// Remove item -----------------------------------------------------------

#[cfg(any(cfail1,cfail4))]
pub trait RemoveItemTrait {
    type TypeName;
    fn method_name();
}

#[cfg(any(cfail1,cfail4))]
impl RemoveItemTrait for Foo {
    type TypeName = ();
    fn method_name() { }
}

#[cfg(not(any(cfail1,cfail4)))]
pub trait RemoveItemTrait {
    type TypeName;
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,associated_item_def_ids", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,associated_item_def_ids", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
impl RemoveItemTrait for Foo {
    type TypeName = ();
}

// Add item -----------------------------------------------------------

#[cfg(any(cfail1,cfail4))]
pub trait AddItemTrait {
    type TypeName;
}

#[cfg(any(cfail1,cfail4))]
impl AddItemTrait for Foo {
    type TypeName = ();
}

#[cfg(not(any(cfail1,cfail4)))]
pub trait AddItemTrait {
    type TypeName;
    fn method_name();
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,associated_item_def_ids", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,associated_item_def_ids", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
impl AddItemTrait for Foo {
    type TypeName = ();
    fn method_name() { }
}

// Change has-value -----------------------------------------------------------

#[cfg(any(cfail1,cfail4))]
pub trait ChangeHasValueTrait {
    //--------------------------------------------------------
    //--------------------------
    //--------------------------------------------------------
    //--------------------------
    fn method_name()   ;
}

#[cfg(any(cfail1,cfail4))]
impl ChangeHasValueTrait for Foo {
    fn method_name() { }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub trait ChangeHasValueTrait {
    #[rustc_clean(except="opt_hir_owner_nodes", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    fn method_name() { }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
impl ChangeHasValueTrait for Foo {
    fn method_name() { }
}

// Add default

pub trait AddDefaultTrait {
    fn method_name();
}

#[cfg(any(cfail1,cfail4))]
impl AddDefaultTrait for Foo {
    // -------------------------------------------------------
    // -------------------------
    // -------------------------------------------------------
    // -------------------------
    fn         method_name() { }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
impl AddDefaultTrait for Foo {
    #[rustc_clean(except="opt_hir_owner_nodes", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    default fn method_name() { }
}

// Add arguments

#[cfg(any(cfail1,cfail4))]
pub trait AddArgumentTrait {
    fn method_name(&self);
}

#[cfg(any(cfail1,cfail4))]
impl AddArgumentTrait for Foo {
    // -----------------------------------------------------------------------------------
    // -------------------------
    // -----------------------------------------------------------------------------------
    // -------------------------
    fn method_name(&self         ) { }
}

#[cfg(not(any(cfail1,cfail4)))]
pub trait AddArgumentTrait {
    fn method_name(&self, x: u32);
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
impl AddArgumentTrait for Foo {
    #[rustc_clean(except="opt_hir_owner_nodes,fn_sig,typeck,optimized_mir", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes,fn_sig,typeck,optimized_mir", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    fn method_name(&self, _x: u32) { }
}

// Change argument type

#[cfg(any(cfail1,cfail4))]
pub trait ChangeArgumentTypeTrait {
    fn method_name(&self, x: u32);
}

#[cfg(any(cfail1,cfail4))]
impl ChangeArgumentTypeTrait for Foo {
    // -----------------------------------------------------------------------------------
    // -------------------------
    // -----------------------------------------------------------------------------------
    // -------------------------
    fn method_name(&self, _x: u32 ) { }
}

#[cfg(not(any(cfail1,cfail4)))]
pub trait ChangeArgumentTypeTrait {
    fn method_name(&self, x: char);
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
impl ChangeArgumentTypeTrait for Foo {
    #[rustc_clean(except="opt_hir_owner_nodes,fn_sig,typeck,optimized_mir", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="opt_hir_owner_nodes,fn_sig,typeck,optimized_mir", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    fn method_name(&self, _x: char) { }
}



struct Bar<T>(T);

// Add Type Parameter To Impl --------------------------------------------------
trait AddTypeParameterToImpl<T> {
    fn id(t: T) -> T;
}

#[cfg(any(cfail1,cfail4))]
impl AddTypeParameterToImpl<u32> for Bar<u32> {
    fn id(t: u32) -> u32 { t }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,generics_of,impl_trait_header", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,generics_of,impl_trait_header", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
impl<TTT> AddTypeParameterToImpl<TTT> for Bar<TTT> {
    #[rustc_clean(
        except="opt_hir_owner_nodes,generics_of,fn_sig,type_of,typeck,optimized_mir",
        cfg="cfail2",
    )]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(
        except="opt_hir_owner_nodes,generics_of,fn_sig,type_of,typeck,optimized_mir",
        cfg="cfail5",
    )]
    #[rustc_clean(cfg="cfail6")]
    fn id(t: TTT) -> TTT { t }
}



// Change Self Type of Impl ----------------------------------------------------
trait ChangeSelfTypeOfImpl {
    fn id(self) -> Self;
}

#[cfg(any(cfail1,cfail4))]
impl ChangeSelfTypeOfImpl for u32 {
    fn id(self) -> Self { self }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes,impl_trait_header", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="opt_hir_owner_nodes,impl_trait_header", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
impl ChangeSelfTypeOfImpl for u64 {
    #[rustc_clean(except="fn_sig,typeck,optimized_mir", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="fn_sig,typeck,optimized_mir", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    fn id(self) -> Self { self }
}



// Add Lifetime Bound to Impl --------------------------------------------------
trait AddLifetimeBoundToImplParameter {
    fn id(self) -> Self;
}

#[cfg(any(cfail1,cfail4))]
impl<T> AddLifetimeBoundToImplParameter for T {
    fn id(self) -> Self { self }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="opt_hir_owner_nodes", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
impl<T: 'static> AddLifetimeBoundToImplParameter for T {
    #[rustc_clean(cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    fn id(self) -> Self { self }
}



// Add Trait Bound to Impl Parameter -------------------------------------------
trait AddTraitBoundToImplParameter {
    fn id(self) -> Self;
}

#[cfg(any(cfail1,cfail4))]
impl<T> AddTraitBoundToImplParameter for T {
    fn id(self) -> Self { self }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="opt_hir_owner_nodes", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="opt_hir_owner_nodes", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
impl<T: Clone> AddTraitBoundToImplParameter for T {
    #[rustc_clean(cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    fn id(self) -> Self { self }
}



// Add #[no_mangle] to Method --------------------------------------------------
trait AddNoMangleToMethod {
    fn add_no_mangle_to_method(&self) { }
}

#[cfg(any(cfail1,cfail4))]
impl AddNoMangleToMethod for Foo {
    // -------------------------
    // -------------------------
    // -------------------------
    // -------------------------
    // ---------
    fn add_no_mangle_to_method(&self) { }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
impl AddNoMangleToMethod for Foo {
    #[rustc_clean(cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    #[no_mangle]
    fn add_no_mangle_to_method(&self) { }
}


// Make Method #[inline] -------------------------------------------------------
trait MakeMethodInline {
    fn make_method_inline(&self) -> u8 { 0 }
}

#[cfg(any(cfail1,cfail4))]
impl MakeMethodInline for Foo {
    // -------------------------
    // -------------------------
    // -------------------------
    // -------------------------
    // ------
    fn make_method_inline(&self) -> u8 { 0 }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
impl MakeMethodInline for Foo {
    #[rustc_clean(cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    #[inline]
    fn make_method_inline(&self) -> u8 { 0 }
}
