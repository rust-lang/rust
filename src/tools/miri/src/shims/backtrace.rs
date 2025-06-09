use rustc_abi::{CanonAbi, FieldIdx, Size};
use rustc_middle::ty::layout::LayoutOf as _;
use rustc_middle::ty::{self, Instance, Ty};
use rustc_span::{BytePos, Loc, Symbol, hygiene};
use rustc_target::callconv::FnAbi;

use crate::*;

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn handle_miri_backtrace_size(
        &mut self,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        link_name: Symbol,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let [flags] = this.check_shim(abi, CanonAbi::Rust, link_name, args)?;

        let flags = this.read_scalar(flags)?.to_u64()?;
        if flags != 0 {
            throw_unsup_format!("unknown `miri_backtrace_size` flags {}", flags);
        }

        let frame_count = this.active_thread_stack().len();

        this.write_scalar(Scalar::from_target_usize(frame_count.to_u64(), this), dest)
    }

    fn handle_miri_get_backtrace(
        &mut self,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        link_name: Symbol,
        args: &[OpTy<'tcx>],
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let ptr_ty = this.machine.layouts.mut_raw_ptr.ty;
        let ptr_layout = this.layout_of(ptr_ty)?;

        let [flags, buf] = this.check_shim(abi, CanonAbi::Rust, link_name, args)?;

        let flags = this.read_scalar(flags)?.to_u64()?;
        let buf_place = this.deref_pointer_as(buf, ptr_layout)?;

        let mut data = Vec::new();
        for frame in this.active_thread_stack().iter().rev() {
            // Match behavior of debuginfo (`FunctionCx::adjusted_span_and_dbg_scope`).
            let span = hygiene::walk_chain_collapsed(frame.current_span(), frame.body().span);
            data.push((frame.instance(), span.lo()));
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
                let fn_ptr = this.fn_ptr(FnVal::Instance(instance));
                fn_ptr.wrapping_offset(Size::from_bytes(pos.0), this)
            })
            .collect();

        match flags {
            0 => {
                throw_unsup_format!("miri_get_backtrace: v0 is not supported any more");
            }
            1 =>
                for (i, ptr) in ptrs.into_iter().enumerate() {
                    let offset = ptr_layout.size.checked_mul(i.to_u64(), this).unwrap();

                    let op_place = buf_place.offset(offset, ptr_layout, this)?;

                    this.write_pointer(ptr, &op_place)?;
                },
            _ => throw_unsup_format!("unknown `miri_get_backtrace` flags {}", flags),
        };

        interp_ok(())
    }

    fn resolve_frame_pointer(
        &mut self,
        ptr: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, (Instance<'tcx>, Loc, String, String)> {
        let this = self.eval_context_mut();

        let ptr = this.read_pointer(ptr)?;
        // Take apart the pointer, we need its pieces. The offset encodes the span.
        let (alloc_id, offset, _prov) = this.ptr_get_alloc_id(ptr, 0)?;

        // This has to be an actual global fn ptr, not a dlsym function.
        let fn_instance = if let Some(GlobalAlloc::Function { instance, .. }) =
            this.tcx.try_get_global_alloc(alloc_id)
        {
            instance
        } else {
            throw_ub_format!("expected static function pointer, found {:?}", ptr);
        };

        let lo =
            this.tcx.sess.source_map().lookup_char_pos(BytePos(offset.bytes().try_into().unwrap()));

        let name = fn_instance.to_string();
        let filename = lo.file.name.prefer_remapped_unconditionaly().to_string();

        interp_ok((fn_instance, lo, name, filename))
    }

    fn handle_miri_resolve_frame(
        &mut self,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        link_name: Symbol,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let [ptr, flags] = this.check_shim(abi, CanonAbi::Rust, link_name, args)?;

        let flags = this.read_scalar(flags)?.to_u64()?;

        let (fn_instance, lo, name, filename) = this.resolve_frame_pointer(ptr)?;

        // Reconstruct the original function pointer,
        // which we pass to user code.
        let fn_ptr = this.fn_ptr(FnVal::Instance(fn_instance));

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

        if let ty::Adt(adt, _) = dest.layout.ty.kind() {
            if !adt.repr().c() {
                throw_ub_format!(
                    "miri_resolve_frame must be declared with a `#[repr(C)]` return type"
                );
            }
        }

        match flags {
            0 => {
                throw_unsup_format!("miri_resolve_frame: v0 is not supported any more");
            }
            1 => {
                this.write_scalar(
                    Scalar::from_target_usize(name.len().to_u64(), this),
                    &this.project_field(dest, FieldIdx::from_u32(0))?,
                )?;
                this.write_scalar(
                    Scalar::from_target_usize(filename.len().to_u64(), this),
                    &this.project_field(dest, FieldIdx::from_u32(1))?,
                )?;
            }
            _ => throw_unsup_format!("unknown `miri_resolve_frame` flags {}", flags),
        }

        this.write_scalar(Scalar::from_u32(lineno), &this.project_field(dest, FieldIdx::from_u32(2))?)?;
        this.write_scalar(Scalar::from_u32(colno), &this.project_field(dest, FieldIdx::from_u32(3))?)?;

        // Support a 4-field struct for now - this is deprecated
        // and slated for removal.
        if num_fields == 5 {
            this.write_pointer(fn_ptr, &this.project_field(dest, FieldIdx::from_u32(4))?)?;
        }

        interp_ok(())
    }

    fn handle_miri_resolve_frame_names(
        &mut self,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        link_name: Symbol,
        args: &[OpTy<'tcx>],
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let [ptr, flags, name_ptr, filename_ptr] =
            this.check_shim(abi, CanonAbi::Rust, link_name, args)?;

        let flags = this.read_scalar(flags)?.to_u64()?;
        if flags != 0 {
            throw_unsup_format!("unknown `miri_resolve_frame_names` flags {}", flags);
        }

        let (_, _, name, filename) = this.resolve_frame_pointer(ptr)?;

        this.write_bytes_ptr(this.read_pointer(name_ptr)?, name.bytes())?;
        this.write_bytes_ptr(this.read_pointer(filename_ptr)?, filename.bytes())?;

        interp_ok(())
    }
}
