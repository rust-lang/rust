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
// for let expressions.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

// must-compile-successfully
// revisions: cfail1 cfail2 cfail3
// compile-flags: -Z query-dep-graph


#![allow(warnings)]
#![feature(rustc_attrs)]
#![feature(specialization)]
#![crate_type="rlib"]

struct Foo;

// Change Method Name -----------------------------------------------------------

#[cfg(cfail1)]
pub trait ChangeMethodNameTrait {
    fn method_name();
}

#[cfg(cfail1)]
impl ChangeMethodNameTrait for Foo {
    fn method_name() { }
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
pub trait ChangeMethodNameTrait {
    #[rustc_dirty(label="Hir", cfg="cfail2")]
    #[rustc_clean(label="Hir", cfg="cfail3")]
    #[rustc_metadata_dirty(cfg="cfail2")]
    #[rustc_metadata_clean(cfg="cfail3")]
    fn method_name2();
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
impl ChangeMethodNameTrait for Foo {
    #[rustc_dirty(label="Hir", cfg="cfail2")]
    #[rustc_clean(label="Hir", cfg="cfail3")]
    #[rustc_metadata_dirty(cfg="cfail2")]
    #[rustc_metadata_clean(cfg="cfail3")]
    fn method_name2() { }
}

// Change Method Body -----------------------------------------------------------
//
// This should affect the method itself, but not the trait.

pub trait ChangeMethodBodyTrait {
    fn method_name();
}

#[cfg(cfail1)]
impl ChangeMethodBodyTrait for Foo {
    fn method_name() { }
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
impl ChangeMethodBodyTrait for Foo {
    #[rustc_dirty(label="Hir", cfg="cfail2")]
    #[rustc_clean(label="Hir", cfg="cfail3")]
    #[rustc_metadata_dirty(cfg="cfail2")]
    #[rustc_metadata_clean(cfg="cfail3")]
    fn method_name() {
        ()
    }
}

// Change Method Selfness -----------------------------------------------------------

#[cfg(cfail1)]
pub trait ChangeMethodSelfnessTrait {
    fn method_name();
}

#[cfg(cfail1)]
impl ChangeMethodSelfnessTrait for Foo {
    fn method_name() { }
}

#[cfg(not(cfail1))]
pub trait ChangeMethodSelfnessTrait {
    fn method_name(&self);
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
impl ChangeMethodSelfnessTrait for Foo {
    #[rustc_dirty(label="Hir", cfg="cfail2")]
    #[rustc_clean(label="Hir", cfg="cfail3")]
    #[rustc_metadata_dirty(cfg="cfail2")]
    #[rustc_metadata_clean(cfg="cfail3")]
    fn method_name(&self) {
        ()
    }
}

// Change Method Selfness -----------------------------------------------------------

#[cfg(cfail1)]
pub trait RemoveMethodSelfnessTrait {
    fn method_name(&self);
}

#[cfg(cfail1)]
impl RemoveMethodSelfnessTrait for Foo {
    fn method_name(&self) { }
}

#[cfg(not(cfail1))]
pub trait RemoveMethodSelfnessTrait {
    fn method_name();
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
impl RemoveMethodSelfnessTrait for Foo {
    #[rustc_dirty(label="Hir", cfg="cfail2")]
    #[rustc_clean(label="Hir", cfg="cfail3")]
    #[rustc_metadata_dirty(cfg="cfail2")]
    #[rustc_metadata_clean(cfg="cfail3")]
    fn method_name() {
        ()
    }
}

// Change Method Selfmutness -----------------------------------------------------------

#[cfg(cfail1)]
pub trait ChangeMethodSelfmutnessTrait {
    fn method_name(&self);
}

#[cfg(cfail1)]
impl ChangeMethodSelfmutnessTrait for Foo {
    fn method_name(&self) { }
}

#[cfg(not(cfail1))]
pub trait ChangeMethodSelfmutnessTrait {
    fn method_name(&mut self);
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
impl ChangeMethodSelfmutnessTrait for Foo {
    #[rustc_dirty(label="Hir", cfg="cfail2")]
    #[rustc_clean(label="Hir", cfg="cfail3")]
    #[rustc_metadata_dirty(cfg="cfail2")]
    #[rustc_metadata_clean(cfg="cfail3")]
    fn method_name(&mut self) {
        ()
    }
}

// Change item kind -----------------------------------------------------------

#[cfg(cfail1)]
pub trait ChangeItemKindTrait {
    fn name();
}

#[cfg(cfail1)]
impl ChangeItemKindTrait for Foo {
    fn name() { }
}

#[cfg(not(cfail1))]
pub trait ChangeItemKindTrait {
    type name;
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
impl ChangeItemKindTrait for Foo {
    type name = ();
}

// Remove item -----------------------------------------------------------

#[cfg(cfail1)]
pub trait RemoveItemTrait {
    type TypeName;
    fn method_name();
}

#[cfg(cfail1)]
impl RemoveItemTrait for Foo {
    type TypeName = ();
    fn method_name() { }
}

#[cfg(not(cfail1))]
pub trait RemoveItemTrait {
    type TypeName;
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
impl RemoveItemTrait for Foo {
    type TypeName = ();
}

// Add item -----------------------------------------------------------

#[cfg(cfail1)]
pub trait AddItemTrait {
    type TypeName;
}

#[cfg(cfail1)]
impl AddItemTrait for Foo {
    type TypeName = ();
}

#[cfg(not(cfail1))]
pub trait AddItemTrait {
    type TypeName;
    fn method_name();
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
impl AddItemTrait for Foo {
    type TypeName = ();
    fn method_name() { }
}

// Change has-value -----------------------------------------------------------

#[cfg(cfail1)]
pub trait ChangeHasValueTrait {
    fn method_name();
}

#[cfg(cfail1)]
impl ChangeHasValueTrait for Foo {
    fn method_name() { }
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
pub trait ChangeHasValueTrait {
    fn method_name() { }
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
impl ChangeHasValueTrait for Foo {
    fn method_name() { }
}

// Add default

pub trait AddDefaultTrait {
    fn method_name();
}

#[cfg(cfail1)]
impl AddDefaultTrait for Foo {
    fn method_name() { }
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
impl AddDefaultTrait for Foo {
    default fn method_name() { }
}

// Remove default

pub trait RemoveDefaultTrait {
    fn method_name();
}

#[cfg(cfail1)]
impl RemoveDefaultTrait for Foo {
    default fn method_name() { }
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
impl RemoveDefaultTrait for Foo {
    fn method_name() { }
}

// Add arguments

#[cfg(cfail1)]
pub trait AddArgumentTrait {
    fn method_name(&self);
}

#[cfg(cfail1)]
impl AddArgumentTrait for Foo {
    fn method_name(&self) { }
}

#[cfg(not(cfail1))]
pub trait AddArgumentTrait {
    fn method_name(&self, x: u32);
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
impl AddArgumentTrait for Foo {
    #[rustc_dirty(label="Hir", cfg="cfail2")]
    #[rustc_clean(label="Hir", cfg="cfail3")]
    #[rustc_metadata_dirty(cfg="cfail2")]
    #[rustc_metadata_clean(cfg="cfail3")]
    fn method_name(&self, _x: u32) { }
}

// Change argument type

#[cfg(cfail1)]
pub trait ChangeArgumentTypeTrait {
    fn method_name(&self, x: u32);
}

#[cfg(cfail1)]
impl ChangeArgumentTypeTrait for Foo {
    fn method_name(&self, _x: u32) { }
}

#[cfg(not(cfail1))]
pub trait ChangeArgumentTypeTrait {
    fn method_name(&self, x: char);
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
impl ChangeArgumentTypeTrait for Foo {
    #[rustc_dirty(label="Hir", cfg="cfail2")]
    #[rustc_clean(label="Hir", cfg="cfail3")]
    #[rustc_metadata_dirty(cfg="cfail2")]
    #[rustc_metadata_clean(cfg="cfail3")]
    fn method_name(&self, _x: char) { }
}



struct Bar<T>(T);

// Add Type Parameter To Impl --------------------------------------------------
trait AddTypeParameterToImpl<T> {
    fn id(t: T) -> T;
}

#[cfg(cfail1)]
impl AddTypeParameterToImpl<u32> for Bar<u32> {
    fn id(t: u32) -> u32 { t }
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
impl<T> AddTypeParameterToImpl<T> for Bar<T> {
    #[rustc_dirty(label="Hir", cfg="cfail2")]
    #[rustc_clean(label="Hir", cfg="cfail3")]
    #[rustc_metadata_dirty(cfg="cfail2")]
    #[rustc_metadata_clean(cfg="cfail3")]
    fn id(t: T) -> T { t }
}



// Change Self Type of Impl ----------------------------------------------------
trait ChangeSelfTypeOfImpl {
    fn id(self) -> Self;
}

#[cfg(cfail1)]
impl ChangeSelfTypeOfImpl for u32 {
    fn id(self) -> Self { self }
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
impl ChangeSelfTypeOfImpl for u64 {
    #[rustc_dirty(label="Hir", cfg="cfail2")]
    #[rustc_clean(label="Hir", cfg="cfail3")]
    #[rustc_metadata_dirty(cfg="cfail2")]
    #[rustc_metadata_clean(cfg="cfail3")]
    fn id(self) -> Self { self }
}



// Add Lifetime Bound to Impl --------------------------------------------------
trait AddLifetimeBoundToImplParameter {
    fn id(self) -> Self;
}

#[cfg(cfail1)]
impl<T> AddLifetimeBoundToImplParameter for T {
    fn id(self) -> Self { self }
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
impl<T: 'static> AddLifetimeBoundToImplParameter for T {
    #[rustc_dirty(label="Hir", cfg="cfail2")]
    #[rustc_clean(label="Hir", cfg="cfail3")]
    #[rustc_metadata_dirty(cfg="cfail2")]
    #[rustc_metadata_clean(cfg="cfail3")]
    fn id(self) -> Self { self }
}



// Add Trait Bound to Impl Parameter -------------------------------------------
trait AddTraitBoundToImplParameter {
    fn id(self) -> Self;
}

#[cfg(cfail1)]
impl<T> AddTraitBoundToImplParameter for T {
    fn id(self) -> Self { self }
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_dirty(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
impl<T: Clone> AddTraitBoundToImplParameter for T {
    #[rustc_dirty(label="Hir", cfg="cfail2")]
    #[rustc_clean(label="Hir", cfg="cfail3")]
    #[rustc_metadata_dirty(cfg="cfail2")]
    #[rustc_metadata_clean(cfg="cfail3")]
    fn id(self) -> Self { self }
}



// Add #[no_mangle] to Method --------------------------------------------------
trait AddNoMangleToMethod {
    fn add_no_mangle_to_method(&self) { }
}

#[cfg(cfail1)]
impl AddNoMangleToMethod for Foo {
    fn add_no_mangle_to_method(&self) { }
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
impl AddNoMangleToMethod for Foo {
    #[rustc_dirty(label="Hir", cfg="cfail2")]
    #[rustc_clean(label="Hir", cfg="cfail3")]
    #[rustc_metadata_dirty(cfg="cfail2")]
    #[rustc_metadata_clean(cfg="cfail3")]
    #[no_mangle]
    fn add_no_mangle_to_method(&self) { }
}


// Make Method #[inline] -------------------------------------------------------
trait MakeMethodInline {
    fn make_method_inline(&self) -> u8 { 0 }
}

#[cfg(cfail1)]
impl MakeMethodInline for Foo {
    fn make_method_inline(&self) -> u8 { 0 }
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
impl MakeMethodInline for Foo {
    #[rustc_dirty(label="Hir", cfg="cfail2")]
    #[rustc_clean(label="Hir", cfg="cfail3")]
    #[rustc_metadata_dirty(cfg="cfail2")]
    #[rustc_metadata_clean(cfg="cfail3")]
    #[inline]
    fn make_method_inline(&self) -> u8 { 0 }
}
