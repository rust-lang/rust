use crate::*;
use rustc_ast::ast::Mutability;
use rustc_middle::ty::layout::LayoutOf as _;
use rustc_middle::ty::{self, Instance, Ty};
use rustc_span::{BytePos, Loc, Symbol};
use rustc_target::{abi::Size, spec::abi::Abi};

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn handle_miri_backtrace_size(
        &mut self,
        abi: Abi,
        link_name: Symbol,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let [flags] = this.check_shim(abi, Abi::Rust, link_name, args)?;

        let flags = this.read_scalar(flags)?.to_u64()?;
        if flags != 0 {
            throw_unsup_format!("unknown `miri_backtrace_size` flags {}", flags);
        }

        let frame_count = this.active_thread_stack().len();

        this.write_scalar(Scalar::from_target_usize(frame_count.try_into().unwrap(), this), dest)
    }

    fn handle_miri_get_backtrace(
        &mut self,
        abi: Abi,
        link_name: Symbol,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let tcx = this.tcx;

        let flags = if let Some(flags_op) = args.get(0) {
            this.read_scalar(flags_op)?.to_u64()?
        } else {
            throw_ub_format!("expected at least 1 argument")
        };

        let mut data = Vec::new();
        for frame in this.active_thread_stack().iter().rev() {
            let mut span = frame.current_span();
            // Match the behavior of runtime backtrace spans
            // by using a non-macro span in our backtrace. See `FunctionCx::debug_loc`.
            if span.from_expansion() && !tcx.sess.opts.unstable_opts.debug_macros {
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
                let fn_ptr = this.create_fn_alloc_ptr(FnVal::Instance(instance));
                fn_ptr.wrapping_offset(Size::from_bytes(pos.0), this)
            })
            .collect();

        let len: u64 = ptrs.len().try_into().unwrap();

        let ptr_ty = this.machine.layouts.mut_raw_ptr.ty;
        let array_layout = this.layout_of(Ty::new_array(tcx.tcx, ptr_ty, len)).unwrap();

        match flags {
            // storage for pointers is allocated by miri
            // deallocating the slice is undefined behavior with a custom global allocator
            0 => {
                let [_flags] = this.check_shim(abi, Abi::Rust, link_name, args)?;

                let alloc = this.allocate(array_layout, MiriMemoryKind::Rust.into())?;

                // Write pointers into array
                for (i, ptr) in ptrs.into_iter().enumerate() {
                    let place = this.mplace_index(&alloc, i as u64)?;

                    this.write_pointer(ptr, &place.into())?;
                }

                this.write_immediate(
                    Immediate::new_slice(Scalar::from_maybe_pointer(alloc.ptr, this), len, this),
                    dest,
                )?;
            }
            // storage for pointers is allocated by the caller
            1 => {
                let [_flags, buf] = this.check_shim(abi, Abi::Rust, link_name, args)?;

                let buf_place = this.deref_operand(buf)?;

                let ptr_layout = this.layout_of(ptr_ty)?;

                for (i, ptr) in ptrs.into_iter().enumerate() {
                    let offset = ptr_layout.size.checked_mul(i.try_into().unwrap(), this).unwrap();

                    let op_place = buf_place.offset(offset, ptr_layout, this)?;

                    this.write_pointer(ptr, &op_place.into())?;
                }
            }
            _ => throw_unsup_format!("unknown `miri_get_backtrace` flags {}", flags),
        };

        Ok(())
    }

    fn resolve_frame_pointer(
        &mut self,
        ptr: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, (Instance<'tcx>, Loc, String, String)> {
        let this = self.eval_context_mut();

        let ptr = this.read_pointer(ptr)?;
        // Take apart the pointer, we need its pieces. The offset encodes the span.
        let (alloc_id, offset, _prov) = this.ptr_get_alloc_id(ptr)?;

        // This has to be an actual global fn ptr, not a dlsym function.
        let fn_instance = if let Some(GlobalAlloc::Function(instance)) =
            this.tcx.try_get_global_alloc(alloc_id)
        {
            instance
        } else {
            throw_ub_format!("expected static function pointer, found {:?}", ptr);
        };

        let lo =
            this.tcx.sess.source_map().lookup_char_pos(BytePos(offset.bytes().try_into().unwrap()));

        let name = fn_instance.to_string();
        let filename = lo.file.name.prefer_remapped().to_string();

        Ok((fn_instance, lo, name, filename))
    }

    fn handle_miri_resolve_frame(
        &mut self,
        abi: Abi,
        link_name: Symbol,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let [ptr, flags] = this.check_shim(abi, Abi::Rust, link_name, args)?;

        let flags = this.read_scalar(flags)?.to_u64()?;

        let (fn_instance, lo, name, filename) = this.resolve_frame_pointer(ptr)?;

        // Reconstruct the original function pointer,
        // which we pass to user code.
        let fn_ptr = this.create_fn_alloc_ptr(FnVal::Instance(fn_instance));

        let num_fields = dest.layout.fields.count();

        if !(4..=5).contains(&num_fields) {
            // Always mention 5 fields, since the 4-field struct
            // is deprecated and slated for removal.
            throw_ub_format!(
                "bad declaration of miri_resolve_frame - should return a struct with 5 fields"
            );
        }

        // `u32` is not enough to fit line/colno, which can be `usize`. It seems unlikely that a
        // file would have more than 2^32 lines or columns, but whatever, just default to 0.
        let lineno: u32 = u32::try_from(lo.line).unwrap_or(0);
        // `lo.col` is 0-based - add 1 to make it 1-based for the caller.
        let colno: u32 = u32::try_from(lo.col.0.saturating_add(1)).unwrap_or(0);

        let dest = this.force_allocation(dest)?;
        if let ty::Adt(adt, _) = dest.layout.ty.kind() {
            if !adt.repr().c() {
                throw_ub_format!(
                    "miri_resolve_frame must be declared with a `#[repr(C)]` return type"
                );
            }
        }

        match flags {
            0 => {
                // These are "mutable" allocations as we consider them to be owned by the callee.
                let name_alloc =
                    this.allocate_str(&name, MiriMemoryKind::Rust.into(), Mutability::Mut)?;
                let filename_alloc =
                    this.allocate_str(&filename, MiriMemoryKind::Rust.into(), Mutability::Mut)?;

                this.write_immediate(
                    name_alloc.to_ref(this),
                    &this.mplace_field(&dest, 0)?.into(),
                )?;
                this.write_immediate(
                    filename_alloc.to_ref(this),
                    &this.mplace_field(&dest, 1)?.into(),
                )?;
            }
            1 => {
                this.write_scalar(
                    Scalar::from_target_usize(name.len().try_into().unwrap(), this),
                    &this.mplace_field(&dest, 0)?.into(),
                )?;
                this.write_scalar(
                    Scalar::from_target_usize(filename.len().try_into().unwrap(), this),
                    &this.mplace_field(&dest, 1)?.into(),
                )?;
            }
            _ => throw_unsup_format!("unknown `miri_resolve_frame` flags {}", flags),
        }

        this.write_scalar(Scalar::from_u32(lineno), &this.mplace_field(&dest, 2)?.into())?;
        this.write_scalar(Scalar::from_u32(colno), &this.mplace_field(&dest, 3)?.into())?;

        // Support a 4-field struct for now - this is deprecated
        // and slated for removal.
        if num_fields == 5 {
            this.write_pointer(fn_ptr, &this.mplace_field(&dest, 4)?.into())?;
        }

        Ok(())
    }

    fn handle_miri_resolve_frame_names(
        &mut self,
        abi: Abi,
        link_name: Symbol,
        args: &[OpTy<'tcx, Provenance>],
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let [ptr, flags, name_ptr, filename_ptr] =
            this.check_shim(abi, Abi::Rust, link_name, args)?;

        let flags = this.read_scalar(flags)?.to_u64()?;
        if flags != 0 {
            throw_unsup_format!("unknown `miri_resolve_frame_names` flags {}", flags);
        }

        let (_, _, name, filename) = this.resolve_frame_pointer(ptr)?;

        this.write_bytes_ptr(this.read_pointer(name_ptr)?, name.bytes())?;
        this.write_bytes_ptr(this.read_pointer(filename_ptr)?, filename.bytes())?;

        Ok(())
    }
}
