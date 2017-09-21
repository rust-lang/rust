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
use abi::FnType;
use callee;
use common::*;
use builder::Builder;
use consts;
use monomorphize;
use type_::Type;
use value::Value;
use rustc::ty::{self, Ty};
use rustc::ty::layout::HasDataLayout;

#[derive(Copy, Clone, Debug)]
pub struct VirtualIndex(u64);

pub const DESTRUCTOR: VirtualIndex = VirtualIndex(0);
pub const SIZE: VirtualIndex = VirtualIndex(1);
pub const ALIGN: VirtualIndex = VirtualIndex(2);

impl<'a, 'tcx> VirtualIndex {
    pub fn from_index(index: usize) -> Self {
        VirtualIndex(index as u64 + 3)
    }

    pub fn get_fn(self, bcx: &Builder<'a, 'tcx>,
                  llvtable: ValueRef,
                  fn_ty: &FnType<'tcx>) -> ValueRef {
        // Load the data pointer from the object.
        debug!("get_fn({:?}, {:?})", Value(llvtable), self);

        let llvtable = bcx.pointercast(llvtable, fn_ty.llvm_type(bcx.ccx).ptr_to().ptr_to());
        let ptr = bcx.load_nonnull(bcx.inbounds_gep(llvtable, &[C_usize(bcx.ccx, self.0)]), None);
        // Vtable loads are invariant
        bcx.set_invariant_load(ptr);
        ptr
    }

    pub fn get_usize(self, bcx: &Builder<'a, 'tcx>, llvtable: ValueRef) -> ValueRef {
        // Load the data pointer from the object.
        debug!("get_int({:?}, {:?})", Value(llvtable), self);

        let llvtable = bcx.pointercast(llvtable, Type::isize(bcx.ccx).ptr_to());
        let ptr = bcx.load(bcx.inbounds_gep(llvtable, &[C_usize(bcx.ccx, self.0)]), None);
        // Vtable loads are invariant
        bcx.set_invariant_load(ptr);
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
pub fn get_vtable<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                            ty: Ty<'tcx>,
                            trait_ref: Option<ty::PolyExistentialTraitRef<'tcx>>)
                            -> ValueRef
{
    let tcx = ccx.tcx();

    debug!("get_vtable(ty={:?}, trait_ref={:?})", ty, trait_ref);

    // Check the cache.
    if let Some(&val) = ccx.vtables().borrow().get(&(ty, trait_ref)) {
        return val;
    }

    // Not in the cache. Build it.
    let nullptr = C_null(Type::i8p(ccx));

    let (size, align) = ccx.size_and_align_of(ty);
    let mut components: Vec<_> = [
        callee::get_fn(ccx, monomorphize::resolve_drop_in_place(ccx.tcx(), ty)),
        C_usize(ccx, size.bytes()),
        C_usize(ccx, align.abi())
    ].iter().cloned().collect();

    if let Some(trait_ref) = trait_ref {
        let trait_ref = trait_ref.with_self_ty(tcx, ty);
        let methods = tcx.vtable_methods(trait_ref);
        let methods = methods.iter().cloned().map(|opt_mth| {
            opt_mth.map_or(nullptr, |(def_id, substs)| {
                callee::resolve_and_get_fn(ccx, def_id, substs)
            })
        });
        components.extend(methods);
    }

    let vtable_const = C_struct(ccx, &components, false);
    let align = ccx.data_layout().pointer_align;
    let vtable = consts::addr_of(ccx, vtable_const, align, "vtable");

    ccx.vtables().borrow_mut().insert((ty, trait_ref), vtable);
    vtable
}
