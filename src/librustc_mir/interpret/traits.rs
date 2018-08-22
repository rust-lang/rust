// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::ty::{self, Ty};
use rustc::ty::layout::{Size, Align, LayoutOf};
use rustc::mir::interpret::{Scalar, Pointer, EvalResult};

use syntax::ast::Mutability;

use super::{EvalContext, Machine, MemoryKind};

impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M> {
    /// Creates a dynamic vtable for the given type and vtable origin. This is used only for
    /// objects.
    ///
    /// The `trait_ref` encodes the erased self type. Hence if we are
    /// making an object `Foo<Trait>` from a value of type `Foo<T>`, then
    /// `trait_ref` would map `T:Trait`.
    pub fn get_vtable(
        &mut self,
        ty: Ty<'tcx>,
        trait_ref: ty::PolyTraitRef<'tcx>,
    ) -> EvalResult<'tcx, Pointer> {
        debug!("get_vtable(trait_ref={:?})", trait_ref);

        let layout = self.layout_of(trait_ref.self_ty())?;
        assert!(!layout.is_unsized(), "can't create a vtable for an unsized type");
        let size = layout.size.bytes();
        let align = layout.align.abi();

        let ptr_size = self.memory.pointer_size();
        let ptr_align = self.tcx.data_layout.pointer_align;
        let methods = self.tcx.vtable_methods(trait_ref);
        let vtable = self.memory.allocate(
            ptr_size * (3 + methods.len() as u64),
            ptr_align,
            MemoryKind::Stack,
        )?;

        let drop = ::monomorphize::resolve_drop_in_place(*self.tcx, ty);
        let drop = self.memory.create_fn_alloc(drop);
        self.memory.write_ptr_sized(vtable, ptr_align, Scalar::Ptr(drop).into())?;

        let size_ptr = vtable.offset(ptr_size, &self)?;
        self.memory.write_ptr_sized(size_ptr, ptr_align, Scalar::Bits {
            bits: size as u128,
            size: ptr_size.bytes() as u8,
        }.into())?;
        let align_ptr = vtable.offset(ptr_size * 2, &self)?;
        self.memory.write_ptr_sized(align_ptr, ptr_align, Scalar::Bits {
            bits: align as u128,
            size: ptr_size.bytes() as u8,
        }.into())?;

        for (i, method) in methods.iter().enumerate() {
            if let Some((def_id, substs)) = *method {
                let instance = self.resolve(def_id, substs)?;
                let fn_ptr = self.memory.create_fn_alloc(instance);
                let method_ptr = vtable.offset(ptr_size * (3 + i as u64), &self)?;
                self.memory.write_ptr_sized(method_ptr, ptr_align, Scalar::Ptr(fn_ptr).into())?;
            }
        }

        self.memory.mark_static_initialized(
            vtable.alloc_id,
            Mutability::Immutable,
        )?;

        Ok(vtable)
    }

    pub fn read_drop_type_from_vtable(
        &self,
        vtable: Pointer,
    ) -> EvalResult<'tcx, ty::Instance<'tcx>> {
        // we don't care about the pointee type, we just want a pointer
        let pointer_align = self.tcx.data_layout.pointer_align;
        let drop_fn = self.memory.read_ptr_sized(vtable, pointer_align)?.to_ptr()?;
        self.memory.get_fn(drop_fn)
    }

    pub fn read_size_and_align_from_vtable(
        &self,
        vtable: Pointer,
    ) -> EvalResult<'tcx, (Size, Align)> {
        let pointer_size = self.memory.pointer_size();
        let pointer_align = self.tcx.data_layout.pointer_align;
        let size = self.memory.read_ptr_sized(vtable.offset(pointer_size, self)?,pointer_align)?
            .to_bits(pointer_size)? as u64;
        let align = self.memory.read_ptr_sized(
            vtable.offset(pointer_size * 2, self)?,
            pointer_align
        )?.to_bits(pointer_size)? as u64;
        Ok((Size::from_bytes(size), Align::from_bytes(align, align).unwrap()))
    }
}
