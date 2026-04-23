// This test case tests the incremental compilation hash (ICH) implementation
// for let expressions.

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
#![feature(specialization)]
#![crate_type = "rlib"]

struct Foo;

// Change Method Name -----------------------------------------------------------

#[cfg(any(bpass1, bpass4))]
pub trait ChangeMethodNameTrait {
    fn method_name();
}

#[cfg(any(bpass1, bpass4))]
impl ChangeMethodNameTrait for Foo {
    fn method_name() {}
}

#[cfg(not(any(bpass1, bpass4)))]
#[rustc_clean(except = "owner,associated_item_def_ids", cfg = "bpass2")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(except = "owner,associated_item_def_ids,predicates_of", cfg = "bpass5")]
#[rustc_clean(cfg = "bpass6")]
pub trait ChangeMethodNameTrait {
    #[rustc_clean(cfg = "bpass3")]
    #[rustc_clean(cfg = "bpass6")]
    fn method_name2();
}

#[cfg(not(any(bpass1, bpass4)))]
#[rustc_clean(except = "owner,associated_item_def_ids", cfg = "bpass2")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(except = "owner,associated_item_def_ids", cfg = "bpass5")]
#[rustc_clean(cfg = "bpass6")]
impl ChangeMethodNameTrait for Foo {
    #[rustc_clean(cfg = "bpass3")]
    #[rustc_clean(cfg = "bpass6")]
    fn method_name2() {}
}

// Change Method Body -----------------------------------------------------------
//
// This should affect the method itself, but not the impl.

pub trait ChangeMethodBodyTrait {
    fn method_name();
}

#[cfg(any(bpass1, bpass4))]
impl ChangeMethodBodyTrait for Foo {
    // --------------------------------------------------------------
    // -------------------------
    // --------------------------------------------------------------
    // -------------------------
    fn method_name() {
        //
    }
}

#[cfg(not(any(bpass1, bpass4)))]
#[rustc_clean(cfg = "bpass2", except = "owner")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(except = "owner", cfg = "bpass5")]
#[rustc_clean(cfg = "bpass6")]
impl ChangeMethodBodyTrait for Foo {
    #[rustc_clean(except = "owner,typeck_root", cfg = "bpass2")]
    #[rustc_clean(cfg = "bpass3")]
    #[rustc_clean(except = "owner,typeck_root", cfg = "bpass5")]
    #[rustc_clean(cfg = "bpass6")]
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

#[cfg(any(bpass1, bpass4))]
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

#[cfg(not(any(bpass1, bpass4)))]
#[rustc_clean(cfg = "bpass2", except = "owner")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(except = "owner", cfg = "bpass5")]
#[rustc_clean(cfg = "bpass6")]
impl ChangeMethodBodyTraitInlined for Foo {
    #[rustc_clean(except = "owner,typeck_root,optimized_mir", cfg = "bpass2")]
    #[rustc_clean(cfg = "bpass3")]
    #[rustc_clean(except = "owner,typeck_root,optimized_mir", cfg = "bpass5")]
    #[rustc_clean(cfg = "bpass6")]
    #[inline]
    fn method_name() {
        panic!()
    }
}

// Change Method Selfness ------------------------------------------------------

#[cfg(any(bpass1, bpass4))]
pub trait ChangeMethodSelfnessTrait {
    fn method_name();
}

#[cfg(any(bpass1, bpass4))]
impl ChangeMethodSelfnessTrait for Foo {
    fn method_name() {}
}

#[cfg(not(any(bpass1, bpass4)))]
pub trait ChangeMethodSelfnessTrait {
    fn method_name(&self);
}

#[cfg(not(any(bpass1, bpass4)))]
#[rustc_clean(cfg = "bpass2", except = "owner")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(except = "owner", cfg = "bpass5")]
#[rustc_clean(cfg = "bpass6")]
impl ChangeMethodSelfnessTrait for Foo {
    #[rustc_clean(
        except = "owner,associated_item,generics_of,fn_sig,typeck_root,optimized_mir",
        cfg = "bpass2"
    )]
    #[rustc_clean(cfg = "bpass3")]
    #[rustc_clean(
        except = "owner,associated_item,generics_of,fn_sig,typeck_root,optimized_mir",
        cfg = "bpass5"
    )]
    #[rustc_clean(cfg = "bpass6")]
    fn method_name(&self) {
        ()
    }
}

// Change Method Selfness -----------------------------------------------------------

#[cfg(any(bpass1, bpass4))]
pub trait RemoveMethodSelfnessTrait {
    fn method_name(&self);
}

#[cfg(any(bpass1, bpass4))]
impl RemoveMethodSelfnessTrait for Foo {
    fn method_name(&self) {}
}

#[cfg(not(any(bpass1, bpass4)))]
pub trait RemoveMethodSelfnessTrait {
    fn method_name();
}

#[cfg(not(any(bpass1, bpass4)))]
#[rustc_clean(cfg = "bpass2", except = "owner")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(except = "owner", cfg = "bpass5")]
#[rustc_clean(cfg = "bpass6")]
impl RemoveMethodSelfnessTrait for Foo {
    #[rustc_clean(
        except = "owner,associated_item,generics_of,fn_sig,typeck_root,optimized_mir",
        cfg = "bpass2"
    )]
    #[rustc_clean(cfg = "bpass3")]
    #[rustc_clean(
        except = "owner,associated_item,generics_of,fn_sig,typeck_root,optimized_mir",
        cfg = "bpass5"
    )]
    #[rustc_clean(cfg = "bpass6")]
    fn method_name() {}
}

// Change Method Selfmutness -----------------------------------------------------------

#[cfg(any(bpass1, bpass4))]
pub trait ChangeMethodSelfmutnessTrait {
    fn method_name(&self);
}

#[cfg(any(bpass1, bpass4))]
impl ChangeMethodSelfmutnessTrait for Foo {
    // -----------------------------------------------------------------------------------
    // -------------------------
    // -----------------------------------------------------------------------------------
    // -------------------------
    fn method_name(&self) {}
}

#[cfg(not(any(bpass1, bpass4)))]
pub trait ChangeMethodSelfmutnessTrait {
    fn method_name(&mut self);
}

#[cfg(not(any(bpass1, bpass4)))]
#[rustc_clean(cfg = "bpass2", except = "owner")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(except = "owner", cfg = "bpass5")]
#[rustc_clean(cfg = "bpass6")]
impl ChangeMethodSelfmutnessTrait for Foo {
    #[rustc_clean(except = "owner,fn_sig,typeck_root,optimized_mir", cfg = "bpass2")]
    #[rustc_clean(cfg = "bpass3")]
    #[rustc_clean(except = "owner,fn_sig,typeck_root,optimized_mir", cfg = "bpass5")]
    #[rustc_clean(cfg = "bpass6")]
    fn method_name(&mut self) {}
}

// Change item kind -----------------------------------------------------------

#[cfg(any(bpass1, bpass4))]
pub trait ChangeItemKindTrait {
    fn name();
}

#[cfg(any(bpass1, bpass4))]
impl ChangeItemKindTrait for Foo {
    fn name() {}
}

#[cfg(not(any(bpass1, bpass4)))]
pub trait ChangeItemKindTrait {
    type name;
}

#[cfg(not(any(bpass1, bpass4)))]
#[rustc_clean(except = "owner,associated_item_def_ids", cfg = "bpass2")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(except = "owner,associated_item_def_ids", cfg = "bpass5")]
#[rustc_clean(cfg = "bpass6")]
impl ChangeItemKindTrait for Foo {
    type name = ();
}

// Remove item -----------------------------------------------------------

#[cfg(any(bpass1, bpass4))]
pub trait RemoveItemTrait {
    type TypeName;
    fn method_name();
}

#[cfg(any(bpass1, bpass4))]
impl RemoveItemTrait for Foo {
    type TypeName = ();
    fn method_name() {}
}

#[cfg(not(any(bpass1, bpass4)))]
pub trait RemoveItemTrait {
    type TypeName;
}

#[cfg(not(any(bpass1, bpass4)))]
#[rustc_clean(except = "owner,associated_item_def_ids", cfg = "bpass2")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(except = "owner,associated_item_def_ids", cfg = "bpass5")]
#[rustc_clean(cfg = "bpass6")]
impl RemoveItemTrait for Foo {
    type TypeName = ();
}

// Add item -----------------------------------------------------------

#[cfg(any(bpass1, bpass4))]
pub trait AddItemTrait {
    type TypeName;
}

#[cfg(any(bpass1, bpass4))]
impl AddItemTrait for Foo {
    type TypeName = ();
}

#[cfg(not(any(bpass1, bpass4)))]
pub trait AddItemTrait {
    type TypeName;
    fn method_name();
}

#[cfg(not(any(bpass1, bpass4)))]
#[rustc_clean(except = "owner,associated_item_def_ids", cfg = "bpass2")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(except = "owner,associated_item_def_ids", cfg = "bpass5")]
#[rustc_clean(cfg = "bpass6")]
impl AddItemTrait for Foo {
    type TypeName = ();
    fn method_name() {}
}

// Change has-value -----------------------------------------------------------

#[cfg(any(bpass1, bpass4))]
pub trait ChangeHasValueTrait {
    //--------------------------------------------------------
    //--------------------------
    //--------------------------------------------------------
    //--------------------------
    fn method_name();
}

#[cfg(any(bpass1, bpass4))]
impl ChangeHasValueTrait for Foo {
    fn method_name() {}
}

#[cfg(not(any(bpass1, bpass4)))]
#[rustc_clean(cfg = "bpass2", except = "owner")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(cfg = "bpass5", except = "owner,predicates_of")]
#[rustc_clean(cfg = "bpass6")]
pub trait ChangeHasValueTrait {
    #[rustc_clean(except = "owner", cfg = "bpass2")]
    #[rustc_clean(cfg = "bpass3")]
    #[rustc_clean(except = "owner", cfg = "bpass5")]
    #[rustc_clean(cfg = "bpass6")]
    fn method_name() {}
}

#[cfg(not(any(bpass1, bpass4)))]
#[rustc_clean(cfg = "bpass2", except = "owner")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(cfg = "bpass5", except = "owner")]
#[rustc_clean(cfg = "bpass6")]
impl ChangeHasValueTrait for Foo {
    fn method_name() {}
}

// Add default

pub trait AddDefaultTrait {
    fn method_name();
}

#[cfg(any(bpass1, bpass4))]
impl AddDefaultTrait for Foo {
    // -------------------------------------------------------
    // -------------------------
    // -------------------------------------------------------
    // -------------------------
    fn method_name() {}
}

#[cfg(not(any(bpass1, bpass4)))]
#[rustc_clean(cfg = "bpass2", except = "owner")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(cfg = "bpass5", except = "owner")]
#[rustc_clean(cfg = "bpass6")]
impl AddDefaultTrait for Foo {
    #[rustc_clean(except = "owner", cfg = "bpass2")]
    #[rustc_clean(cfg = "bpass3")]
    #[rustc_clean(except = "owner,optimized_mir ", cfg = "bpass5")]
    #[rustc_clean(cfg = "bpass6")]
    default fn method_name() {}
}

// Add arguments

#[cfg(any(bpass1, bpass4))]
pub trait AddArgumentTrait {
    fn method_name(&self);
}

#[cfg(any(bpass1, bpass4))]
impl AddArgumentTrait for Foo {
    // -----------------------------------------------------------------------------------
    // -------------------------
    // -----------------------------------------------------------------------------------
    // -------------------------
    fn method_name(&self) {}
}

#[cfg(not(any(bpass1, bpass4)))]
pub trait AddArgumentTrait {
    fn method_name(&self, x: u32);
}

#[cfg(not(any(bpass1, bpass4)))]
#[rustc_clean(cfg = "bpass2", except = "owner")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(except = "owner", cfg = "bpass5")]
#[rustc_clean(cfg = "bpass6")]
impl AddArgumentTrait for Foo {
    #[rustc_clean(except = "owner,fn_sig,typeck_root,optimized_mir", cfg = "bpass2")]
    #[rustc_clean(cfg = "bpass3")]
    #[rustc_clean(except = "owner,fn_sig,typeck_root,optimized_mir", cfg = "bpass5")]
    #[rustc_clean(cfg = "bpass6")]
    fn method_name(&self, _x: u32) {}
}

// Change argument type

#[cfg(any(bpass1, bpass4))]
pub trait ChangeArgumentTypeTrait {
    fn method_name(&self, x: u32);
}

#[cfg(any(bpass1, bpass4))]
impl ChangeArgumentTypeTrait for Foo {
    // -----------------------------------------------------------------------------------
    // -------------------------
    // -----------------------------------------------------------------------------------
    // -------------------------
    fn method_name(&self, _x: u32) {}
}

#[cfg(not(any(bpass1, bpass4)))]
pub trait ChangeArgumentTypeTrait {
    fn method_name(&self, x: char);
}

#[cfg(not(any(bpass1, bpass4)))]
#[rustc_clean(cfg = "bpass2", except = "owner")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(except = "owner", cfg = "bpass5")]
#[rustc_clean(cfg = "bpass6")]
impl ChangeArgumentTypeTrait for Foo {
    #[rustc_clean(except = "owner,fn_sig,typeck_root,optimized_mir", cfg = "bpass2")]
    #[rustc_clean(cfg = "bpass3")]
    #[rustc_clean(except = "owner,fn_sig,typeck_root,optimized_mir", cfg = "bpass5")]
    #[rustc_clean(cfg = "bpass6")]
    fn method_name(&self, _x: char) {}
}

struct Bar<T>(T);

// Add Type Parameter To Impl --------------------------------------------------
trait AddTypeParameterToImpl<T> {
    fn id(t: T) -> T;
}

#[cfg(any(bpass1, bpass4))]
impl AddTypeParameterToImpl<u32> for Bar<u32> {
    fn id(t: u32) -> u32 {
        t
    }
}

#[cfg(not(any(bpass1, bpass4)))]
#[rustc_clean(except = "owner,generics_of,impl_trait_header", cfg = "bpass2")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(except = "owner,generics_of,impl_trait_header", cfg = "bpass5")]
#[rustc_clean(cfg = "bpass6")]
impl<TTT> AddTypeParameterToImpl<TTT> for Bar<TTT> {
    #[rustc_clean(
        except = "owner,generics_of,fn_sig,type_of,typeck_root,optimized_mir",
        cfg = "bpass2"
    )]
    #[rustc_clean(cfg = "bpass3")]
    #[rustc_clean(
        except = "owner,generics_of,fn_sig,type_of,typeck_root,optimized_mir",
        cfg = "bpass5"
    )]
    #[rustc_clean(cfg = "bpass6")]
    fn id(t: TTT) -> TTT {
        t
    }
}

// Change Self Type of Impl ----------------------------------------------------
trait ChangeSelfTypeOfImpl {
    fn id(self) -> Self;
}

#[cfg(any(bpass1, bpass4))]
impl ChangeSelfTypeOfImpl for u32 {
    fn id(self) -> Self {
        self
    }
}

#[cfg(not(any(bpass1, bpass4)))]
#[rustc_clean(except = "owner,impl_trait_header", cfg = "bpass2")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(except = "owner,impl_trait_header", cfg = "bpass5")]
#[rustc_clean(cfg = "bpass6")]
impl ChangeSelfTypeOfImpl for u64 {
    #[rustc_clean(except = "fn_sig,typeck_root,optimized_mir,owner", cfg = "bpass2")]
    #[rustc_clean(cfg = "bpass3")]
    #[rustc_clean(except = "fn_sig,typeck_root,optimized_mir,owner", cfg = "bpass5")]
    #[rustc_clean(cfg = "bpass6")]
    fn id(self) -> Self {
        self
    }
}

// Add Lifetime Bound to Impl --------------------------------------------------
trait AddLifetimeBoundToImplParameter {
    fn id(self) -> Self;
}

#[cfg(any(bpass1, bpass4))]
impl<T> AddLifetimeBoundToImplParameter for T {
    fn id(self) -> Self {
        self
    }
}

#[cfg(not(any(bpass1, bpass4)))]
#[rustc_clean(except = "owner", cfg = "bpass2")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(except = "owner", cfg = "bpass5")]
#[rustc_clean(cfg = "bpass6")]
impl<T: 'static> AddLifetimeBoundToImplParameter for T {
    #[rustc_clean(cfg = "bpass2", except = "owner")]
    #[rustc_clean(cfg = "bpass3")]
    #[rustc_clean(cfg = "bpass5", except = "owner")]
    #[rustc_clean(cfg = "bpass6")]
    fn id(self) -> Self {
        self
    }
}

// Add Trait Bound to Impl Parameter -------------------------------------------
trait AddTraitBoundToImplParameter {
    fn id(self) -> Self;
}

#[cfg(any(bpass1, bpass4))]
impl<T> AddTraitBoundToImplParameter for T {
    fn id(self) -> Self {
        self
    }
}

#[cfg(not(any(bpass1, bpass4)))]
#[rustc_clean(except = "owner", cfg = "bpass2")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(except = "owner", cfg = "bpass5")]
#[rustc_clean(cfg = "bpass6")]
impl<T: Clone> AddTraitBoundToImplParameter for T {
    #[rustc_clean(cfg = "bpass2", except = "owner")]
    #[rustc_clean(cfg = "bpass3")]
    #[rustc_clean(cfg = "bpass5", except = "owner")]
    #[rustc_clean(cfg = "bpass6")]
    fn id(self) -> Self {
        self
    }
}

// Add #[no_mangle] to Method --------------------------------------------------
trait AddNoMangleToMethod {
    fn add_no_mangle_to_method(&self) {}
}

#[cfg(any(bpass1, bpass4))]
impl AddNoMangleToMethod for Foo {
    // -------------------------
    // -------------------------
    // -------------------------
    // -------------------------
    // -----------------
    fn add_no_mangle_to_method(&self) {}
}

#[cfg(not(any(bpass1, bpass4)))]
#[rustc_clean(cfg = "bpass2", except = "owner")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(cfg = "bpass5", except = "owner")]
#[rustc_clean(cfg = "bpass6")]
impl AddNoMangleToMethod for Foo {
    #[rustc_clean(cfg = "bpass2", except = "owner")]
    #[rustc_clean(cfg = "bpass3")]
    #[rustc_clean(cfg = "bpass5", except = "owner")]
    #[rustc_clean(cfg = "bpass6")]
    #[unsafe(no_mangle)]
    fn add_no_mangle_to_method(&self) {}
}

// Make Method #[inline] -------------------------------------------------------
trait MakeMethodInline {
    fn make_method_inline(&self) -> u8 {
        0
    }
}

#[cfg(any(bpass1, bpass4))]
impl MakeMethodInline for Foo {
    // -------------------------
    // -------------------------
    // -------------------------
    // -------------------------
    // ------
    fn make_method_inline(&self) -> u8 {
        0
    }
}

#[cfg(not(any(bpass1, bpass4)))]
#[rustc_clean(cfg = "bpass2", except = "owner")]
#[rustc_clean(cfg = "bpass3")]
#[rustc_clean(cfg = "bpass5", except = "owner")]
#[rustc_clean(cfg = "bpass6")]
impl MakeMethodInline for Foo {
    #[rustc_clean(cfg = "bpass2", except = "owner")]
    #[rustc_clean(cfg = "bpass3")]
    #[rustc_clean(cfg = "bpass5", except = "owner")]
    #[rustc_clean(cfg = "bpass6")]
    #[inline]
    fn make_method_inline(&self) -> u8 {
        0
    }
}
