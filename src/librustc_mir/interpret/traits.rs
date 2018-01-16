use rustc::ty::{self, Ty};
use rustc::ty::layout::{Size, Align, LayoutOf};
use syntax::ast::Mutability;

use rustc::mir::interpret::{PrimVal, Value, MemoryPointer, EvalResult};
use super::{EvalContext, eval_context,
            Machine};

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
    ) -> EvalResult<'tcx, MemoryPointer> {
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
            None,
        )?;

        let drop = eval_context::resolve_drop_in_place(self.tcx, ty);
        let drop = self.memory.create_fn_alloc(drop);
        self.memory.write_ptr_sized_unsigned(vtable, ptr_align, PrimVal::Ptr(drop))?;

        let size_ptr = vtable.offset(ptr_size, &self)?;
        self.memory.write_ptr_sized_unsigned(size_ptr, ptr_align, PrimVal::Bytes(size as u128))?;
        let align_ptr = vtable.offset(ptr_size * 2, &self)?;
        self.memory.write_ptr_sized_unsigned(align_ptr, ptr_align, PrimVal::Bytes(align as u128))?;

        for (i, method) in methods.iter().enumerate() {
            if let Some((def_id, substs)) = *method {
                let instance = self.resolve(def_id, substs)?;
                let fn_ptr = self.memory.create_fn_alloc(instance);
                let method_ptr = vtable.offset(ptr_size * (3 + i as u64), &self)?;
                self.memory.write_ptr_sized_unsigned(method_ptr, ptr_align, PrimVal::Ptr(fn_ptr))?;
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
        vtable: MemoryPointer,
    ) -> EvalResult<'tcx, Option<ty::Instance<'tcx>>> {
        // we don't care about the pointee type, we just want a pointer
        let pointer_align = self.tcx.data_layout.pointer_align;
        match self.read_ptr(vtable, pointer_align, self.tcx.mk_nil_ptr())? {
            // some values don't need to call a drop impl, so the value is null
            Value::ByVal(PrimVal::Bytes(0)) => Ok(None),
            Value::ByVal(PrimVal::Ptr(drop_fn)) => self.memory.get_fn(drop_fn).map(Some),
            _ => err!(ReadBytesAsPointer),
        }
    }

    pub fn read_size_and_align_from_vtable(
        &self,
        vtable: MemoryPointer,
    ) -> EvalResult<'tcx, (Size, Align)> {
        let pointer_size = self.memory.pointer_size();
        let pointer_align = self.tcx.data_layout.pointer_align;
        let size = self.memory.read_ptr_sized_unsigned(vtable.offset(pointer_size, self)?, pointer_align)?.to_bytes()? as u64;
        let align = self.memory.read_ptr_sized_unsigned(
            vtable.offset(pointer_size * 2, self)?,
            pointer_align
        )?.to_bytes()? as u64;
        Ok((Size::from_bytes(size), Align::from_bytes(align, align).unwrap()))
    }
}
