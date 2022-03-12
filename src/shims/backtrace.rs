use crate::*;
use rustc_ast::ast::Mutability;
use rustc_middle::ty::layout::LayoutOf as _;
use rustc_middle::ty::{self, TypeAndMut};
use rustc_span::{BytePos, Symbol};
use rustc_target::{abi::Size, spec::abi::Abi};
use std::convert::TryInto as _;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn handle_miri_get_backtrace(
        &mut self,
        abi: Abi,
        link_name: Symbol,
        args: &[OpTy<'tcx, Tag>],
        dest: &PlaceTy<'tcx, Tag>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let tcx = this.tcx;
        let &[ref flags] = this.check_shim(abi, Abi::Rust, link_name, args)?;

        let flags = this.read_scalar(flags)?.to_u64()?;
        if flags != 0 {
            throw_unsup_format!("unknown `miri_get_backtrace` flags {}", flags);
        }

        let mut data = Vec::new();
        for frame in this.active_thread_stack().iter().rev() {
            let mut span = frame.current_span();
            // Match the behavior of runtime backtrace spans
            // by using a non-macro span in our backtrace. See `FunctionCx::debug_loc`.
            if span.from_expansion() && !tcx.sess.opts.debugging_opts.debug_macros {
                span = rustc_span::hygiene::walk_chain(span, frame.body.span.ctxt())
            }
            data.push((frame.instance, span.lo()));
        }

        let ptrs: Vec<_> = data
            .into_iter()
            .map(|(instance, pos)| {
                // We represent a frame pointer by using the `span.lo` value
                // as an offset into the function's allocation. This gives us an
                // opaque pointer that we can return to user code, and allows us
                // to reconstruct the needed frame information in `handle_miri_resolve_frame`.
                // Note that we never actually read or write anything from/to this pointer -
                // all of the data is represented by the pointer value itself.
                let fn_ptr = this.memory.create_fn_alloc(FnVal::Instance(instance));
                fn_ptr.wrapping_offset(Size::from_bytes(pos.0), this)
            })
            .collect();

        let len = ptrs.len();

        let ptr_ty = tcx.mk_ptr(TypeAndMut { ty: tcx.types.unit, mutbl: Mutability::Mut });

        let array_ty = tcx.mk_array(ptr_ty, ptrs.len().try_into().unwrap());

        // Write pointers into array
        let alloc =
            this.allocate(this.layout_of(array_ty).unwrap(), MiriMemoryKind::Rust.into())?;
        for (i, ptr) in ptrs.into_iter().enumerate() {
            let place = this.mplace_index(&alloc, i as u64)?;
            this.write_pointer(ptr, &place.into())?;
        }

        this.write_immediate(
            Immediate::new_slice(
                Scalar::from_maybe_pointer(alloc.ptr, this),
                len.try_into().unwrap(),
                this,
            ),
            dest,
        )?;
        Ok(())
    }

    fn handle_miri_resolve_frame(
        &mut self,
        abi: Abi,
        link_name: Symbol,
        args: &[OpTy<'tcx, Tag>],
        dest: &PlaceTy<'tcx, Tag>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let tcx = this.tcx;
        let &[ref ptr, ref flags] = this.check_shim(abi, Abi::Rust, link_name, args)?;

        let flags = this.read_scalar(flags)?.to_u64()?;
        if flags != 0 {
            throw_unsup_format!("unknown `miri_resolve_frame` flags {}", flags);
        }

        let ptr = this.read_pointer(ptr)?;
        // Take apart the pointer, we need its pieces.
        let (alloc_id, offset, ptr) = this.memory.ptr_get_alloc(ptr)?;

        let fn_instance =
            if let Some(GlobalAlloc::Function(instance)) = this.tcx.get_global_alloc(alloc_id) {
                instance
            } else {
                throw_ub_format!("expected function pointer, found {:?}", ptr);
            };

        // Reconstruct the original function pointer,
        // which we pass to user code.
        let fn_ptr = this.memory.create_fn_alloc(FnVal::Instance(fn_instance));

        let num_fields = dest.layout.fields.count();

        if !(4..=5).contains(&num_fields) {
            // Always mention 5 fields, since the 4-field struct
            // is deprecated and slated for removal.
            throw_ub_format!(
                "bad declaration of miri_resolve_frame - should return a struct with 5 fields"
            );
        }

        let pos = BytePos(offset.bytes().try_into().unwrap());
        let name = fn_instance.to_string();

        let lo = tcx.sess.source_map().lookup_char_pos(pos);

        let filename = lo.file.name.prefer_remapped().to_string();
        let lineno: u32 = lo.line as u32;
        // `lo.col` is 0-based - add 1 to make it 1-based for the caller.
        let colno: u32 = lo.col.0 as u32 + 1;

        // These are "mutable" allocations as we consider them to be owned by the callee.
        let name_alloc = this.allocate_str(&name, MiriMemoryKind::Rust.into(), Mutability::Mut);
        let filename_alloc =
            this.allocate_str(&filename, MiriMemoryKind::Rust.into(), Mutability::Mut);
        let lineno_alloc = Scalar::from_u32(lineno);
        let colno_alloc = Scalar::from_u32(colno);

        let dest = this.force_allocation(dest)?;
        if let ty::Adt(adt, _) = dest.layout.ty.kind() {
            if !adt.repr().c() {
                throw_ub_format!(
                    "miri_resolve_frame must be declared with a `#[repr(C)]` return type"
                );
            }
        }

        this.write_immediate(name_alloc.to_ref(this), &this.mplace_field(&dest, 0)?.into())?;
        this.write_immediate(filename_alloc.to_ref(this), &this.mplace_field(&dest, 1)?.into())?;
        this.write_scalar(lineno_alloc, &this.mplace_field(&dest, 2)?.into())?;
        this.write_scalar(colno_alloc, &this.mplace_field(&dest, 3)?.into())?;

        // Support a 4-field struct for now - this is deprecated
        // and slated for removal.
        if num_fields == 5 {
            this.write_pointer(fn_ptr, &this.mplace_field(&dest, 4)?.into())?;
        }

        Ok(())
    }
}
