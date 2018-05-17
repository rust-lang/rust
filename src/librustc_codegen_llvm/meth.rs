// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm::ValueRef;
use abi::{FnType, FnTypeExt};
use callee;
use common::*;
use builder::Builder;
use consts;
use monomorphize;
use type_::Type;
use value::Value;
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

    pub fn get_fn(self, bx: &Builder<'a, 'tcx>,
                  llvtable: ValueRef,
                  fn_ty: &FnType<'tcx, Ty<'tcx>>) -> ValueRef {
        // Load the data pointer from the object.
        debug!("get_fn({:?}, {:?})", Value(llvtable), self);

        let llvtable = bx.pointercast(llvtable, fn_ty.llvm_type(bx.cx).ptr_to().ptr_to());
        let ptr_align = bx.tcx().data_layout.pointer_align;
        let ptr = bx.load(bx.inbounds_gep(llvtable, &[C_usize(bx.cx, self.0)]), ptr_align);
        bx.nonnull_metadata(ptr);
        // Vtable loads are invariant
        bx.set_invariant_load(ptr);
        ptr
    }

    pub fn get_usize(self, bx: &Builder<'a, 'tcx>, llvtable: ValueRef) -> ValueRef {
        // Load the data pointer from the object.
        debug!("get_int({:?}, {:?})", Value(llvtable), self);

        let llvtable = bx.pointercast(llvtable, Type::isize(bx.cx).ptr_to());
        let usize_align = bx.tcx().data_layout.pointer_align;
        let ptr = bx.load(bx.inbounds_gep(llvtable, &[C_usize(bx.cx, self.0)]), usize_align);
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
pub fn get_vtable<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                            ty: Ty<'tcx>,
                            trait_ref: Option<ty::PolyExistentialTraitRef<'tcx>>)
                            -> ValueRef
{
    let tcx = cx.tcx;

    debug!("get_vtable(ty={:?}, trait_ref={:?})", ty, trait_ref);

    // Check the cache.
    if let Some(&val) = cx.vtables.borrow().get(&(ty, trait_ref)) {
        return val;
    }

    // Not in the cache. Build it.
    let nullptr = C_null(Type::i8p(cx));

    let (size, align) = cx.size_and_align_of(ty);
    let mut components: Vec<_> = [
        callee::get_fn(cx, monomorphize::resolve_drop_in_place(cx.tcx, ty)),
        C_usize(cx, size.bytes()),
        C_usize(cx, align.abi())
    ].iter().cloned().collect();

    if let Some(trait_ref) = trait_ref {
        let trait_ref = trait_ref.with_self_ty(tcx, ty);
        let methods = tcx.vtable_methods(trait_ref);
        let methods = methods.iter().cloned().map(|opt_mth| {
            opt_mth.map_or(nullptr, |(def_id, substs)| {
                callee::resolve_and_get_fn(cx, def_id, substs)
            })
        });
        components.extend(methods);
    }

    let vtable_const = C_struct(cx, &components, false);
    let align = cx.data_layout().pointer_align;
    let vtable = consts::addr_of(cx, vtable_const, align, "vtable");

    debuginfo::create_vtable_metadata(cx, ty, vtable);

    cx.vtables.borrow_mut().insert((ty, trait_ref), vtable);
    vtable
}
