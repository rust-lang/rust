use rustc_data_structures::sync::Lrc;
use rustc::ty::{self, Ty};
use rustc::ty::layout::{Size, Align, LayoutOf};
use rustc::mir::interpret::{Scalar, Pointer, EvalResult, PointerArithmetic};

use super::{EvalContext, Machine, MemoryKind};

impl<'a, 'mir, 'tcx, M: Machine<'a, 'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M> {
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
    ) -> EvalResult<'tcx, Pointer<M::PointerTag>> {
        trace!("get_vtable(trait_ref={:?})", poly_trait_ref);

        let (ty, poly_trait_ref) = self.tcx.erase_regions(&(ty, poly_trait_ref));

        if let Some(&vtable) = self.vtables.get(&(ty, poly_trait_ref)) {
            // This means we guarantee that there are no duplicate vtables, we will
            // always use the same vtable for the same (Type, Trait) combination.
            // That's not what happens in rustc, but emulating per-crate deduplication
            // does not sound like it actually makes anything any better.
            return Ok(Pointer::from(vtable).with_default_tag());
        }

        let methods = if let Some(poly_trait_ref) = poly_trait_ref {
            let trait_ref = poly_trait_ref.with_self_ty(*self.tcx, ty);
            let trait_ref = self.tcx.erase_regions(&trait_ref);

            self.tcx.vtable_methods(trait_ref)
        } else {
            Lrc::new(Vec::new())
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
        ).with_default_tag();
        let tcx = &*self.tcx;

        let drop = crate::monomorphize::resolve_drop_in_place(*tcx, ty);
        let drop = self.memory.create_fn_alloc(drop).with_default_tag();
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
                let instance = self.resolve(def_id, substs)?;
                let fn_ptr = self.memory.create_fn_alloc(instance).with_default_tag();
                let method_ptr = vtable.offset(ptr_size * (3 + i as u64), self)?;
                self.memory
                    .get_mut(method_ptr.alloc_id)?
                    .write_ptr_sized(tcx, method_ptr, Scalar::Ptr(fn_ptr).into())?;
            }
        }

        self.memory.mark_immutable(vtable.alloc_id)?;
        assert!(self.vtables.insert((ty, poly_trait_ref), vtable.alloc_id).is_none());

        Ok(vtable)
    }

    /// Returns the drop fn instance as well as the actual dynamic type
    pub fn read_drop_type_from_vtable(
        &self,
        vtable: Pointer<M::PointerTag>,
    ) -> EvalResult<'tcx, (ty::Instance<'tcx>, ty::Ty<'tcx>)> {
        // we don't care about the pointee type, we just want a pointer
        self.memory.check_align(vtable.into(), self.tcx.data_layout.pointer_align.abi)?;
        let drop_fn = self.memory
            .get(vtable.alloc_id)?
            .read_ptr_sized(self, vtable)?
            .to_ptr()?;
        let drop_instance = self.memory.get_fn(drop_fn)?;
        trace!("Found drop fn: {:?}", drop_instance);
        let fn_sig = drop_instance.ty(*self.tcx).fn_sig(*self.tcx);
        let fn_sig = self.tcx.normalize_erasing_late_bound_regions(self.param_env, &fn_sig);
        // the drop function takes *mut T where T is the type being dropped, so get that
        let ty = fn_sig.inputs()[0].builtin_deref(true).unwrap().ty;
        Ok((drop_instance, ty))
    }

    pub fn read_size_and_align_from_vtable(
        &self,
        vtable: Pointer<M::PointerTag>,
    ) -> EvalResult<'tcx, (Size, Align)> {
        let pointer_size = self.pointer_size();
        self.memory.check_align(vtable.into(), self.tcx.data_layout.pointer_align.abi)?;
        let alloc = self.memory.get(vtable.alloc_id)?;
        let size = alloc.read_ptr_sized(self, vtable.offset(pointer_size, self)?)?
            .to_bits(pointer_size)? as u64;
        let align = alloc.read_ptr_sized(
            self,
            vtable.offset(pointer_size * 2, self)?,
        )?.to_bits(pointer_size)? as u64;
        Ok((Size::from_bytes(size), Align::from_bytes(align).unwrap()))
    }
}
