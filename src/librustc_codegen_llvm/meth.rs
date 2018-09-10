// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use abi::{FnType, FnTypeExt};
use callee;
use context::CodegenCx;
use builder::Builder;
use monomorphize;
use value::Value;

use interfaces::{BuilderMethods, ConstMethods, BaseTypeMethods, DerivedTypeMethods, StaticMethods};

use rustc::ty::{self, Ty};
use rustc::ty::layout::HasDataLayout;
use debuginfo;

#[derive(Copy, Clone, Debug)]
pub struct VirtualIndex(u64);

pub const DESTRUCTOR: VirtualIndex = VirtualIndex(0);
pub const SIZE: VirtualIndex = VirtualIndex(1);
pub const ALIGN: VirtualIndex = VirtualIndex(2);

impl<'a, 'tcx> VirtualIndex {
    pub fn from_index(index: usize) -> Self {
        VirtualIndex(index as u64 + 3)
    }

    pub fn get_fn(self, bx: &Builder<'a, 'll, 'tcx>,
                  llvtable: &'ll Value,
                  fn_ty: &FnType<'tcx, Ty<'tcx>>) -> &'ll Value {
        // Load the data pointer from the object.
        debug!("get_fn({:?}, {:?})", llvtable, self);

        let llvtable = bx.pointercast(
            llvtable,
            bx.cx().type_ptr_to(fn_ty.ptr_to_llvm_type(bx.cx()))
        );
        let ptr_align = bx.tcx().data_layout.pointer_align;
        let ptr = bx.load(
            bx.inbounds_gep(llvtable, &[bx.cx().const_usize(self.0)]),
            ptr_align
        );
        bx.nonnull_metadata(ptr);
        // Vtable loads are invariant
        bx.set_invariant_load(ptr);
        ptr
    }

    pub fn get_usize(
        self,
        bx: &Builder<'a, 'll, 'tcx>,
        llvtable: &'ll Value
    ) -> &'ll Value {
        // Load the data pointer from the object.
        debug!("get_int({:?}, {:?})", llvtable, self);

        let llvtable = bx.pointercast(llvtable, bx.cx().type_ptr_to(bx.cx().type_isize()));
        let usize_align = bx.tcx().data_layout.pointer_align;
        let ptr = bx.load(
            bx.inbounds_gep(llvtable, &[bx.cx().const_usize(self.0)]),
            usize_align
        );
        // Vtable loads are invariant
        bx.set_invariant_load(ptr);
        ptr
    }
}

/// Creates a dynamic vtable for the given type and vtable origin.
/// This is used only for objects.
///
/// The vtables are cached instead of created on every call.
///
/// The `trait_ref` encodes the erased self type. Hence if we are
/// making an object `Foo<Trait>` from a value of type `Foo<T>`, then
/// `trait_ref` would map `T:Trait`.
pub fn get_vtable(
    cx: &CodegenCx<'ll, 'tcx>,
    ty: Ty<'tcx>,
    trait_ref: ty::PolyExistentialTraitRef<'tcx>,
) -> &'ll Value {
    let tcx = cx.tcx;

    debug!("get_vtable(ty={:?}, trait_ref={:?})", ty, trait_ref);

    // Check the cache.
    if let Some(&val) = cx.vtables.borrow().get(&(ty, trait_ref)) {
        return val;
    }

    // Not in the cache. Build it.
    let nullptr = cx.const_null(cx.type_i8p());

    let methods = tcx.vtable_methods(trait_ref.with_self_ty(tcx, ty));
    let methods = methods.iter().cloned().map(|opt_mth| {
        opt_mth.map_or(nullptr, |(def_id, substs)| {
            callee::resolve_and_get_fn_for_vtable(cx, def_id, substs)
        })
    });

    let (size, align) = cx.size_and_align_of(ty);
    // /////////////////////////////////////////////////////////////////////////////////////////////
    // If you touch this code, be sure to also make the corresponding changes to
    // `get_vtable` in rust_mir/interpret/traits.rs
    // /////////////////////////////////////////////////////////////////////////////////////////////
    let components: Vec<_> = [
        callee::get_fn(cx, monomorphize::resolve_drop_in_place(cx.tcx, ty)),
        cx.const_usize(size.bytes()),
        cx.const_usize(align.abi())
    ].iter().cloned().chain(methods).collect();

    let vtable_const = cx.const_struct(&components, false);
    let align = cx.data_layout().pointer_align;
    let vtable = cx.static_addr_of(vtable_const, align, Some("vtable"));

    debuginfo::create_vtable_metadata(cx, ty, vtable);

    cx.vtables.borrow_mut().insert((ty, trait_ref), vtable);
    vtable
}
