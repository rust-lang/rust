use rustc::ty::{self, Ty, Instance};
use rustc::ty::layout::{Size, Align, LayoutOf};
use rustc::mir::interpret::{Scalar, Pointer, InterpResult, PointerArithmetic};

use super::{InterpCx, InterpError, Machine, MemoryKind};

impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
    /// Creates a dynamic vtable for the given type and vtable origin. This is used only for
    /// objects.
    ///
    /// The `trait_ref` encodes the erased self type. Hence if we are
    /// making an object `Foo<Trait>` from a value of type `Foo<T>`, then
    /// `trait_ref` would map `T:Trait`.
    pub fn get_vtable(
        &mut self,
        ty: Ty<'tcx>,
        poly_trait_ref: Option<ty::PolyExistentialTraitRef<'tcx>>,
    ) -> InterpResult<'tcx, Pointer<M::PointerTag>> {
        trace!("get_vtable(trait_ref={:?})", poly_trait_ref);

        let (ty, poly_trait_ref) = self.tcx.erase_regions(&(ty, poly_trait_ref));

        if let Some(&vtable) = self.vtables.get(&(ty, poly_trait_ref)) {
            // This means we guarantee that there are no duplicate vtables, we will
            // always use the same vtable for the same (Type, Trait) combination.
            // That's not what happens in rustc, but emulating per-crate deduplication
            // does not sound like it actually makes anything any better.
            return Ok(vtable);
        }

        let methods = if let Some(poly_trait_ref) = poly_trait_ref {
            let trait_ref = poly_trait_ref.with_self_ty(*self.tcx, ty);
            let trait_ref = self.tcx.erase_regions(&trait_ref);

            self.tcx.vtable_methods(trait_ref)
        } else {
            &[]
        };

        let layout = self.layout_of(ty)?;
        assert!(!layout.is_unsized(), "can't create a vtable for an unsized type");
        let size = layout.size.bytes();
        let align = layout.align.abi.bytes();

        let ptr_size = self.pointer_size();
        let ptr_align = self.tcx.data_layout.pointer_align.abi;
        // /////////////////////////////////////////////////////////////////////////////////////////
        // If you touch this code, be sure to also make the corresponding changes to
        // `get_vtable` in rust_codegen_llvm/meth.rs
        // /////////////////////////////////////////////////////////////////////////////////////////
        let vtable = self.memory.allocate(
            ptr_size * (3 + methods.len() as u64),
            ptr_align,
            MemoryKind::Vtable,
        );
        let tcx = &*self.tcx;

        let drop = Instance::resolve_drop_in_place(*tcx, ty);
        let drop = self.memory.create_fn_alloc(drop);

        // no need to do any alignment checks on the memory accesses below, because we know the
        // allocation is correctly aligned as we created it above. Also we're only offsetting by
        // multiples of `ptr_align`, which means that it will stay aligned to `ptr_align`.
        self.memory
            .get_mut(vtable.alloc_id)?
            .write_ptr_sized(tcx, vtable, Scalar::Ptr(drop).into())?;

        let size_ptr = vtable.offset(ptr_size, self)?;
        self.memory
            .get_mut(size_ptr.alloc_id)?
            .write_ptr_sized(tcx, size_ptr, Scalar::from_uint(size, ptr_size).into())?;
        let align_ptr = vtable.offset(ptr_size * 2, self)?;
        self.memory
            .get_mut(align_ptr.alloc_id)?
            .write_ptr_sized(tcx, align_ptr, Scalar::from_uint(align, ptr_size).into())?;

        for (i, method) in methods.iter().enumerate() {
            if let Some((def_id, substs)) = *method {
                // resolve for vtable: insert shims where needed
                let substs = self.subst_and_normalize_erasing_regions(substs)?;
                let instance = ty::Instance::resolve_for_vtable(
                    *self.tcx,
                    self.param_env,
                    def_id,
                    substs,
                ).ok_or_else(|| InterpError::TooGeneric)?;
                let fn_ptr = self.memory.create_fn_alloc(instance);
                let method_ptr = vtable.offset(ptr_size * (3 + i as u64), self)?;
                self.memory
                    .get_mut(method_ptr.alloc_id)?
                    .write_ptr_sized(tcx, method_ptr, Scalar::Ptr(fn_ptr).into())?;
            }
        }

        self.memory.mark_immutable(vtable.alloc_id)?;
        assert!(self.vtables.insert((ty, poly_trait_ref), vtable).is_none());

        Ok(vtable)
    }

    /// Returns the drop fn instance as well as the actual dynamic type
    pub fn read_drop_type_from_vtable(
        &self,
        vtable: Scalar<M::PointerTag>,
    ) -> InterpResult<'tcx, (ty::Instance<'tcx>, Ty<'tcx>)> {
        // we don't care about the pointee type, we just want a pointer
        let vtable = self.memory.check_ptr_access(
            vtable,
            self.tcx.data_layout.pointer_size,
            self.tcx.data_layout.pointer_align.abi,
        )?.expect("cannot be a ZST");
        let drop_fn = self.memory
            .get(vtable.alloc_id)?
            .read_ptr_sized(self, vtable)?
            .to_ptr()?;
        let drop_instance = self.memory.get_fn(drop_fn)?;
        trace!("Found drop fn: {:?}", drop_instance);
        let fn_sig = drop_instance.ty(*self.tcx).fn_sig(*self.tcx);
        let fn_sig = self.tcx.normalize_erasing_late_bound_regions(self.param_env, &fn_sig);
        // The drop function takes `*mut T` where `T` is the type being dropped, so get that.
        let ty = fn_sig.inputs()[0].builtin_deref(true).unwrap().ty;
        Ok((drop_instance, ty))
    }

    pub fn read_size_and_align_from_vtable(
        &self,
        vtable: Scalar<M::PointerTag>,
    ) -> InterpResult<'tcx, (Size, Align)> {
        let pointer_size = self.pointer_size();
        // We check for size = 3*ptr_size, that covers the drop fn (unused here),
        // the size, and the align (which we read below).
        let vtable = self.memory.check_ptr_access(
            vtable,
            3*pointer_size,
            self.tcx.data_layout.pointer_align.abi,
        )?.expect("cannot be a ZST");
        let alloc = self.memory.get(vtable.alloc_id)?;
        let size = alloc.read_ptr_sized(
            self,
            vtable.offset(pointer_size, self)?
        )?.to_bits(pointer_size)? as u64;
        let align = alloc.read_ptr_sized(
            self,
            vtable.offset(pointer_size * 2, self)?,
        )?.to_bits(pointer_size)? as u64;
        Ok((Size::from_bytes(size), Align::from_bytes(align).unwrap()))
    }
}
