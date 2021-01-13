use std::convert::TryFrom;

use rustc_middle::mir::interpret::{InterpResult, Pointer, PointerArithmetic, Scalar};
use rustc_middle::ty::{self, Instance, Ty};
use rustc_target::abi::{Align, LayoutOf, Size};

use super::util::ensure_monomorphic_enough;
use super::{FnVal, InterpCx, Machine, MemoryKind};

impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
    /// Creates a dynamic vtable for the given type and vtable origin. This is used only for
    /// objects.
    ///
    /// The `trait_ref` encodes the erased self type. Hence, if we are
    /// making an object `Foo<Trait>` from a value of type `Foo<T>`, then
    /// `trait_ref` would map `T: Trait`.
    pub fn get_vtable(
        &mut self,
        ty: Ty<'tcx>,
        poly_trait_ref: Option<ty::PolyExistentialTraitRef<'tcx>>,
    ) -> InterpResult<'tcx, Pointer<M::PointerTag>> {
        trace!("get_vtable(trait_ref={:?})", poly_trait_ref);

        let (ty, poly_trait_ref) = self.tcx.erase_regions((ty, poly_trait_ref));

        // All vtables must be monomorphic, bail out otherwise.
        ensure_monomorphic_enough(*self.tcx, ty)?;
        ensure_monomorphic_enough(*self.tcx, poly_trait_ref)?;

        if let Some(&vtable) = self.vtables.get(&(ty, poly_trait_ref)) {
            // This means we guarantee that there are no duplicate vtables, we will
            // always use the same vtable for the same (Type, Trait) combination.
            // That's not what happens in rustc, but emulating per-crate deduplication
            // does not sound like it actually makes anything any better.
            return Ok(vtable);
        }

        let methods = if let Some(poly_trait_ref) = poly_trait_ref {
            let trait_ref = poly_trait_ref.with_self_ty(*self.tcx, ty);
            let trait_ref = self.tcx.erase_regions(trait_ref);

            self.tcx.vtable_methods(trait_ref)
        } else {
            &[]
        };

        let layout = self.layout_of(ty)?;
        assert!(!layout.is_unsized(), "can't create a vtable for an unsized type");
        let size = layout.size.bytes();
        let align = layout.align.abi.bytes();

        let tcx = *self.tcx;
        let ptr_size = self.pointer_size();
        let ptr_align = tcx.data_layout.pointer_align.abi;
        // /////////////////////////////////////////////////////////////////////////////////////////
        // If you touch this code, be sure to also make the corresponding changes to
        // `get_vtable` in `rust_codegen_llvm/meth.rs`.
        // /////////////////////////////////////////////////////////////////////////////////////////
        let vtable_size = ptr_size * u64::try_from(methods.len()).unwrap().checked_add(3).unwrap();
        let vtable = self.memory.allocate(vtable_size, ptr_align, MemoryKind::Vtable);

        let drop = Instance::resolve_drop_in_place(tcx, ty);
        let drop = self.memory.create_fn_alloc(FnVal::Instance(drop));

        // No need to do any alignment checks on the memory accesses below, because we know the
        // allocation is correctly aligned as we created it above. Also we're only offsetting by
        // multiples of `ptr_align`, which means that it will stay aligned to `ptr_align`.
        let vtable_alloc = self.memory.get_raw_mut(vtable.alloc_id)?;
        vtable_alloc.write_ptr_sized(&tcx, vtable, drop.into())?;

        let size_ptr = vtable.offset(ptr_size, &tcx)?;
        vtable_alloc.write_ptr_sized(&tcx, size_ptr, Scalar::from_uint(size, ptr_size).into())?;
        let align_ptr = vtable.offset(ptr_size * 2, &tcx)?;
        vtable_alloc.write_ptr_sized(&tcx, align_ptr, Scalar::from_uint(align, ptr_size).into())?;

        for (i, method) in methods.iter().enumerate() {
            if let Some((def_id, substs)) = *method {
                // resolve for vtable: insert shims where needed
                let instance =
                    ty::Instance::resolve_for_vtable(tcx, self.param_env, def_id, substs)
                        .ok_or_else(|| err_inval!(TooGeneric))?;
                let fn_ptr = self.memory.create_fn_alloc(FnVal::Instance(instance));
                // We cannot use `vtable_allic` as we are creating fn ptrs in this loop.
                let method_ptr = vtable.offset(ptr_size * (3 + i as u64), &tcx)?;
                self.memory.get_raw_mut(vtable.alloc_id)?.write_ptr_sized(
                    &tcx,
                    method_ptr,
                    fn_ptr.into(),
                )?;
            }
        }

        M::after_static_mem_initialized(self, vtable, vtable_size)?;

        self.memory.mark_immutable(vtable.alloc_id)?;
        assert!(self.vtables.insert((ty, poly_trait_ref), vtable).is_none());

        Ok(vtable)
    }

    /// Resolves the function at the specified slot in the provided
    /// vtable. An index of '0' corresponds to the first method
    /// declared in the trait of the provided vtable.
    pub fn get_vtable_slot(
        &self,
        vtable: Scalar<M::PointerTag>,
        idx: u64,
    ) -> InterpResult<'tcx, FnVal<'tcx, M::ExtraFnVal>> {
        let ptr_size = self.pointer_size();
        // Skip over the 'drop_ptr', 'size', and 'align' fields.
        let vtable_slot = vtable.ptr_offset(ptr_size * idx.checked_add(3).unwrap(), self)?;
        let vtable_slot = self
            .memory
            .check_ptr_access(vtable_slot, ptr_size, self.tcx.data_layout.pointer_align.abi)?
            .expect("cannot be a ZST");
        let fn_ptr = self
            .memory
            .get_raw(vtable_slot.alloc_id)?
            .read_ptr_sized(self, vtable_slot)?
            .check_init()?;
        Ok(self.memory.get_fn(fn_ptr)?)
    }

    /// Returns the drop fn instance as well as the actual dynamic type.
    pub fn read_drop_type_from_vtable(
        &self,
        vtable: Scalar<M::PointerTag>,
    ) -> InterpResult<'tcx, (ty::Instance<'tcx>, Ty<'tcx>)> {
        // We don't care about the pointee type; we just want a pointer.
        let vtable = self
            .memory
            .check_ptr_access(
                vtable,
                self.tcx.data_layout.pointer_size,
                self.tcx.data_layout.pointer_align.abi,
            )?
            .expect("cannot be a ZST");
        let drop_fn =
            self.memory.get_raw(vtable.alloc_id)?.read_ptr_sized(self, vtable)?.check_init()?;
        // We *need* an instance here, no other kind of function value, to be able
        // to determine the type.
        let drop_instance = self.memory.get_fn(drop_fn)?.as_instance()?;
        trace!("Found drop fn: {:?}", drop_instance);
        let fn_sig = drop_instance.ty(*self.tcx, self.param_env).fn_sig(*self.tcx);
        let fn_sig = self.tcx.normalize_erasing_late_bound_regions(self.param_env, fn_sig);
        // The drop function takes `*mut T` where `T` is the type being dropped, so get that.
        let args = fn_sig.inputs();
        if args.len() != 1 {
            throw_ub!(InvalidDropFn(fn_sig));
        }
        let ty = args[0].builtin_deref(true).ok_or_else(|| err_ub!(InvalidDropFn(fn_sig)))?.ty;
        Ok((drop_instance, ty))
    }

    pub fn read_size_and_align_from_vtable(
        &self,
        vtable: Scalar<M::PointerTag>,
    ) -> InterpResult<'tcx, (Size, Align)> {
        let pointer_size = self.pointer_size();
        // We check for `size = 3 * ptr_size`, which covers the drop fn (unused here),
        // the size, and the align (which we read below).
        let vtable = self
            .memory
            .check_ptr_access(vtable, 3 * pointer_size, self.tcx.data_layout.pointer_align.abi)?
            .expect("cannot be a ZST");
        let alloc = self.memory.get_raw(vtable.alloc_id)?;
        let size = alloc.read_ptr_sized(self, vtable.offset(pointer_size, self)?)?.check_init()?;
        let size = u64::try_from(self.force_bits(size, pointer_size)?).unwrap();
        let align =
            alloc.read_ptr_sized(self, vtable.offset(pointer_size * 2, self)?)?.check_init()?;
        let align = u64::try_from(self.force_bits(align, pointer_size)?).unwrap();

        if size >= self.tcx.data_layout.obj_size_bound() {
            throw_ub_format!(
                "invalid vtable: \
                size is bigger than largest supported object"
            );
        }
        Ok((Size::from_bytes(size), Align::from_bytes(align).unwrap()))
    }
}
