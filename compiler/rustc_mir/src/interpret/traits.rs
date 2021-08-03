use std::convert::TryFrom;

use rustc_middle::mir::interpret::{InterpResult, Pointer, PointerArithmetic};
use rustc_middle::ty::{
    self, Ty, COMMON_VTABLE_ENTRIES, COMMON_VTABLE_ENTRIES_ALIGN,
    COMMON_VTABLE_ENTRIES_DROPINPLACE, COMMON_VTABLE_ENTRIES_SIZE,
};
use rustc_target::abi::{Align, Size};

use super::util::ensure_monomorphic_enough;
use super::{FnVal, InterpCx, Machine};

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
    ) -> InterpResult<'tcx, Pointer<Option<M::PointerTag>>> {
        trace!("get_vtable(trait_ref={:?})", poly_trait_ref);

        let (ty, poly_trait_ref) = self.tcx.erase_regions((ty, poly_trait_ref));

        // All vtables must be monomorphic, bail out otherwise.
        ensure_monomorphic_enough(*self.tcx, ty)?;
        ensure_monomorphic_enough(*self.tcx, poly_trait_ref)?;

        let vtable_allocation = self.tcx.vtable_allocation(ty, poly_trait_ref);

        let vtable_ptr = self.memory.global_base_pointer(Pointer::from(vtable_allocation))?;

        Ok(vtable_ptr.into())
    }

    /// Resolves the function at the specified slot in the provided
    /// vtable. Currently an index of '3' (`COMMON_VTABLE_ENTRIES.len()`)
    /// corresponds to the first method declared in the trait of the provided vtable.
    pub fn get_vtable_slot(
        &self,
        vtable: Pointer<Option<M::PointerTag>>,
        idx: u64,
    ) -> InterpResult<'tcx, FnVal<'tcx, M::ExtraFnVal>> {
        let ptr_size = self.pointer_size();
        let vtable_slot = vtable.offset(ptr_size * idx, self)?;
        let vtable_slot = self
            .memory
            .get(vtable_slot, ptr_size, self.tcx.data_layout.pointer_align.abi)?
            .expect("cannot be a ZST");
        let fn_ptr = self.scalar_to_ptr(vtable_slot.read_ptr_sized(Size::ZERO)?.check_init()?);
        self.memory.get_fn(fn_ptr)
    }

    /// Returns the drop fn instance as well as the actual dynamic type.
    pub fn read_drop_type_from_vtable(
        &self,
        vtable: Pointer<Option<M::PointerTag>>,
    ) -> InterpResult<'tcx, (ty::Instance<'tcx>, Ty<'tcx>)> {
        let pointer_size = self.pointer_size();
        // We don't care about the pointee type; we just want a pointer.
        let vtable = self
            .memory
            .get(
                vtable,
                pointer_size * u64::try_from(COMMON_VTABLE_ENTRIES.len()).unwrap(),
                self.tcx.data_layout.pointer_align.abi,
            )?
            .expect("cannot be a ZST");
        let drop_fn = vtable
            .read_ptr_sized(
                pointer_size * u64::try_from(COMMON_VTABLE_ENTRIES_DROPINPLACE).unwrap(),
            )?
            .check_init()?;
        // We *need* an instance here, no other kind of function value, to be able
        // to determine the type.
        let drop_instance = self.memory.get_fn(self.scalar_to_ptr(drop_fn))?.as_instance()?;
        trace!("Found drop fn: {:?}", drop_instance);
        let fn_sig = drop_instance.ty(*self.tcx, self.param_env).fn_sig(*self.tcx);
        let fn_sig = self.tcx.normalize_erasing_late_bound_regions(self.param_env, fn_sig);
        // The drop function takes `*mut T` where `T` is the type being dropped, so get that.
        let args = fn_sig.inputs();
        if args.len() != 1 {
            throw_ub!(InvalidVtableDropFn(fn_sig));
        }
        let ty =
            args[0].builtin_deref(true).ok_or_else(|| err_ub!(InvalidVtableDropFn(fn_sig)))?.ty;
        Ok((drop_instance, ty))
    }

    pub fn read_size_and_align_from_vtable(
        &self,
        vtable: Pointer<Option<M::PointerTag>>,
    ) -> InterpResult<'tcx, (Size, Align)> {
        let pointer_size = self.pointer_size();
        // We check for `size = 3 * ptr_size`, which covers the drop fn (unused here),
        // the size, and the align (which we read below).
        let vtable = self
            .memory
            .get(
                vtable,
                pointer_size * u64::try_from(COMMON_VTABLE_ENTRIES.len()).unwrap(),
                self.tcx.data_layout.pointer_align.abi,
            )?
            .expect("cannot be a ZST");
        let size = vtable
            .read_ptr_sized(pointer_size * u64::try_from(COMMON_VTABLE_ENTRIES_SIZE).unwrap())?
            .check_init()?;
        let size = size.to_machine_usize(self)?;
        let align = vtable
            .read_ptr_sized(pointer_size * u64::try_from(COMMON_VTABLE_ENTRIES_ALIGN).unwrap())?
            .check_init()?;
        let align = align.to_machine_usize(self)?;
        let align = Align::from_bytes(align).map_err(|e| err_ub!(InvalidVtableAlignment(e)))?;

        if size >= self.tcx.data_layout.obj_size_bound() {
            throw_ub!(InvalidVtableSize);
        }
        Ok((Size::from_bytes(size), align))
    }

    pub fn read_new_vtable_after_trait_upcasting_from_vtable(
        &self,
        vtable: Pointer<Option<M::PointerTag>>,
        idx: u64,
    ) -> InterpResult<'tcx, Pointer<Option<M::PointerTag>>> {
        let pointer_size = self.pointer_size();

        let vtable_slot = vtable.offset(pointer_size * idx, self)?;
        let new_vtable = self
            .memory
            .get(vtable_slot, pointer_size, self.tcx.data_layout.pointer_align.abi)?
            .expect("cannot be a ZST");

        let new_vtable = self.scalar_to_ptr(new_vtable.read_ptr_sized(Size::ZERO)?.check_init()?);

        Ok(new_vtable)
    }
}
