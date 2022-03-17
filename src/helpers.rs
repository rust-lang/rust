use std::convert::{TryFrom, TryInto};
use std::mem;
use std::num::NonZeroUsize;
use std::time::Duration;

use log::trace;

use rustc_hir::def_id::{DefId, CRATE_DEF_INDEX};
use rustc_middle::mir;
use rustc_middle::ty::{
    self,
    layout::{LayoutOf, TyAndLayout},
    List, TyCtxt,
};
use rustc_span::Symbol;
use rustc_target::abi::{Align, FieldsShape, Size, Variants};
use rustc_target::spec::abi::Abi;

use rand::RngCore;

use crate::*;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}

/// Gets an instance for a path.
fn try_resolve_did<'mir, 'tcx>(tcx: TyCtxt<'tcx>, path: &[&str]) -> Option<DefId> {
    tcx.crates(()).iter().find(|&&krate| tcx.crate_name(krate).as_str() == path[0]).and_then(
        |krate| {
            let krate = DefId { krate: *krate, index: CRATE_DEF_INDEX };
            let mut items = tcx.module_children(krate);
            let mut path_it = path.iter().skip(1).peekable();

            while let Some(segment) = path_it.next() {
                for item in mem::replace(&mut items, Default::default()).iter() {
                    if item.ident.name.as_str() == *segment {
                        if path_it.peek().is_none() {
                            return Some(item.res.def_id());
                        }

                        items = tcx.module_children(item.res.def_id());
                        break;
                    }
                }
            }
            None
        },
    )
}

pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    /// Gets an instance for a path.
    fn resolve_path(&self, path: &[&str]) -> ty::Instance<'tcx> {
        let did = try_resolve_did(self.eval_context_ref().tcx.tcx, path)
            .unwrap_or_else(|| panic!("failed to find required Rust item: {:?}", path));
        ty::Instance::mono(self.eval_context_ref().tcx.tcx, did)
    }

    /// Evaluates the scalar at the specified path. Returns Some(val)
    /// if the path could be resolved, and None otherwise
    fn eval_path_scalar(&self, path: &[&str]) -> InterpResult<'tcx, Scalar<Tag>> {
        let this = self.eval_context_ref();
        let instance = this.resolve_path(path);
        let cid = GlobalId { instance, promoted: None };
        let const_val = this.eval_to_allocation(cid)?;
        let const_val = this.read_scalar(&const_val.into())?;
        return Ok(const_val.check_init()?);
    }

    /// Helper function to get a `libc` constant as a `Scalar`.
    fn eval_libc(&self, name: &str) -> InterpResult<'tcx, Scalar<Tag>> {
        self.eval_path_scalar(&["libc", name])
    }

    /// Helper function to get a `libc` constant as an `i32`.
    fn eval_libc_i32(&self, name: &str) -> InterpResult<'tcx, i32> {
        // TODO: Cache the result.
        self.eval_libc(name)?.to_i32()
    }

    /// Helper function to get a `windows` constant as a `Scalar`.
    fn eval_windows(&self, module: &str, name: &str) -> InterpResult<'tcx, Scalar<Tag>> {
        self.eval_context_ref().eval_path_scalar(&["std", "sys", "windows", module, name])
    }

    /// Helper function to get a `windows` constant as a `u64`.
    fn eval_windows_u64(&self, module: &str, name: &str) -> InterpResult<'tcx, u64> {
        // TODO: Cache the result.
        self.eval_windows(module, name)?.to_u64()
    }

    /// Helper function to get the `TyAndLayout` of a `libc` type
    fn libc_ty_layout(&self, name: &str) -> InterpResult<'tcx, TyAndLayout<'tcx>> {
        let this = self.eval_context_ref();
        let ty = this.resolve_path(&["libc", name]).ty(*this.tcx, ty::ParamEnv::reveal_all());
        this.layout_of(ty)
    }

    /// Helper function to get the `TyAndLayout` of a `windows` type
    fn windows_ty_layout(&self, name: &str) -> InterpResult<'tcx, TyAndLayout<'tcx>> {
        let this = self.eval_context_ref();
        let ty = this
            .resolve_path(&["std", "sys", "windows", "c", name])
            .ty(*this.tcx, ty::ParamEnv::reveal_all());
        this.layout_of(ty)
    }

    /// Project to the given *named* field of the mplace (which must be a struct or union type).
    fn mplace_field_named(
        &self,
        mplace: &MPlaceTy<'tcx, Tag>,
        name: &str,
    ) -> InterpResult<'tcx, MPlaceTy<'tcx, Tag>> {
        let this = self.eval_context_ref();
        let adt = mplace.layout.ty.ty_adt_def().unwrap();
        for (idx, field) in adt.non_enum_variant().fields.iter().enumerate() {
            if field.name.as_str() == name {
                return this.mplace_field(mplace, idx);
            }
        }
        bug!("No field named {} in type {}", name, mplace.layout.ty);
    }

    /// Write an int of the appropriate size to `dest`. The target type may be signed or unsigned,
    /// we try to do the right thing anyway. `i128` can fit all integer types except for `u128` so
    /// this method is fine for almost all integer types.
    fn write_int(&mut self, i: impl Into<i128>, dest: &PlaceTy<'tcx, Tag>) -> InterpResult<'tcx> {
        assert!(dest.layout.abi.is_scalar(), "write_int on non-scalar type {}", dest.layout.ty);
        let val = if dest.layout.abi.is_signed() {
            Scalar::from_int(i, dest.layout.size)
        } else {
            Scalar::from_uint(u64::try_from(i.into()).unwrap(), dest.layout.size)
        };
        self.eval_context_mut().write_scalar(val, dest)
    }

    /// Write the first N fields of the given place.
    fn write_int_fields(
        &mut self,
        values: &[i128],
        dest: &MPlaceTy<'tcx, Tag>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        for (idx, &val) in values.iter().enumerate() {
            let field = this.mplace_field(dest, idx)?;
            this.write_int(val, &field.into())?;
        }
        Ok(())
    }

    /// Write the given fields of the given place.
    fn write_int_fields_named(
        &mut self,
        values: &[(&str, i128)],
        dest: &MPlaceTy<'tcx, Tag>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        for &(name, val) in values.iter() {
            let field = this.mplace_field_named(dest, name)?;
            this.write_int(val, &field.into())?;
        }
        Ok(())
    }

    /// Write a 0 of the appropriate size to `dest`.
    fn write_null(&mut self, dest: &PlaceTy<'tcx, Tag>) -> InterpResult<'tcx> {
        self.write_int(0, dest)
    }

    /// Test if this pointer equals 0.
    fn ptr_is_null(&self, ptr: Pointer<Option<Tag>>) -> InterpResult<'tcx, bool> {
        let this = self.eval_context_ref();
        let null = Scalar::null_ptr(this);
        this.ptr_eq(Scalar::from_maybe_pointer(ptr, this), null)
    }

    /// Get the `Place` for a local
    fn local_place(&mut self, local: mir::Local) -> InterpResult<'tcx, PlaceTy<'tcx, Tag>> {
        let this = self.eval_context_mut();
        let place = mir::Place { local: local, projection: List::empty() };
        this.eval_place(place)
    }

    /// Generate some random bytes, and write them to `dest`.
    fn gen_random(&mut self, ptr: Pointer<Option<Tag>>, len: u64) -> InterpResult<'tcx> {
        // Some programs pass in a null pointer and a length of 0
        // to their platform's random-generation function (e.g. getrandom())
        // on Linux. For compatibility with these programs, we don't perform
        // any additional checks - it's okay if the pointer is invalid,
        // since we wouldn't actually be writing to it.
        if len == 0 {
            return Ok(());
        }
        let this = self.eval_context_mut();

        let mut data = vec![0; usize::try_from(len).unwrap()];

        if this.machine.communicate() {
            // Fill the buffer using the host's rng.
            getrandom::getrandom(&mut data)
                .map_err(|err| err_unsup_format!("host getrandom failed: {}", err))?;
        } else {
            let rng = this.memory.extra.rng.get_mut();
            rng.fill_bytes(&mut data);
        }

        this.memory.write_bytes(ptr, data.iter().copied())
    }

    /// Call a function: Push the stack frame and pass the arguments.
    /// For now, arguments must be scalars (so that the caller does not have to know the layout).
    fn call_function(
        &mut self,
        f: ty::Instance<'tcx>,
        caller_abi: Abi,
        args: &[Immediate<Tag>],
        dest: Option<&PlaceTy<'tcx, Tag>>,
        stack_pop: StackPopCleanup,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let param_env = ty::ParamEnv::reveal_all(); // in Miri this is always the param_env we use... and this.param_env is private.
        let callee_abi = f.ty(*this.tcx, param_env).fn_sig(*this.tcx).abi();
        if this.machine.enforce_abi && callee_abi != caller_abi {
            throw_ub_format!(
                "calling a function with ABI {} using caller ABI {}",
                callee_abi.name(),
                caller_abi.name()
            )
        }

        // Push frame.
        let mir = &*this.load_mir(f.def, None)?;
        this.push_stack_frame(f, mir, dest, stack_pop)?;

        // Initialize arguments.
        let mut callee_args = this.frame().body.args_iter();
        for arg in args {
            let callee_arg = this.local_place(
                callee_args
                    .next()
                    .ok_or_else(|| err_ub_format!("callee has fewer arguments than expected"))?,
            )?;
            this.write_immediate(*arg, &callee_arg)?;
        }
        if callee_args.next().is_some() {
            throw_ub_format!("callee has more arguments than expected");
        }

        Ok(())
    }

    /// Visits the memory covered by `place`, sensitive to freezing: the 2nd parameter
    /// of `action` will be true if this is frozen, false if this is in an `UnsafeCell`.
    /// The range is relative to `place`.
    ///
    /// Assumes that the `place` has a proper pointer in it.
    fn visit_freeze_sensitive(
        &self,
        place: &MPlaceTy<'tcx, Tag>,
        size: Size,
        mut action: impl FnMut(AllocRange, bool) -> InterpResult<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_ref();
        trace!("visit_frozen(place={:?}, size={:?})", *place, size);
        debug_assert_eq!(
            size,
            this.size_and_align_of_mplace(place)?
                .map(|(size, _)| size)
                .unwrap_or_else(|| place.layout.size)
        );
        // Store how far we proceeded into the place so far. Everything to the left of
        // this offset has already been handled, in the sense that the frozen parts
        // have had `action` called on them.
        let ptr = place.ptr.into_pointer_or_addr().unwrap();
        let start_offset = ptr.into_parts().1 as Size; // we just compare offsets, the abs. value never matters
        let mut cur_offset = start_offset;
        // Called when we detected an `UnsafeCell` at the given offset and size.
        // Calls `action` and advances `cur_ptr`.
        let mut unsafe_cell_action = |unsafe_cell_ptr: Pointer<Option<Tag>>,
                                      unsafe_cell_size: Size| {
            let unsafe_cell_ptr = unsafe_cell_ptr.into_pointer_or_addr().unwrap();
            debug_assert_eq!(unsafe_cell_ptr.provenance, ptr.provenance);
            // We assume that we are given the fields in increasing offset order,
            // and nothing else changes.
            let unsafe_cell_offset = unsafe_cell_ptr.into_parts().1 as Size; // we just compare offsets, the abs. value never matters
            assert!(unsafe_cell_offset >= cur_offset);
            let frozen_size = unsafe_cell_offset - cur_offset;
            // Everything between the cur_ptr and this `UnsafeCell` is frozen.
            if frozen_size != Size::ZERO {
                action(alloc_range(cur_offset - start_offset, frozen_size), /*frozen*/ true)?;
            }
            cur_offset += frozen_size;
            // This `UnsafeCell` is NOT frozen.
            if unsafe_cell_size != Size::ZERO {
                action(
                    alloc_range(cur_offset - start_offset, unsafe_cell_size),
                    /*frozen*/ false,
                )?;
            }
            cur_offset += unsafe_cell_size;
            // Done
            Ok(())
        };
        // Run a visitor
        {
            let mut visitor = UnsafeCellVisitor {
                ecx: this,
                unsafe_cell_action: |place| {
                    trace!("unsafe_cell_action on {:?}", place.ptr);
                    // We need a size to go on.
                    let unsafe_cell_size = this
                        .size_and_align_of_mplace(&place)?
                        .map(|(size, _)| size)
                        // for extern types, just cover what we can
                        .unwrap_or_else(|| place.layout.size);
                    // Now handle this `UnsafeCell`, unless it is empty.
                    if unsafe_cell_size != Size::ZERO {
                        unsafe_cell_action(place.ptr, unsafe_cell_size)
                    } else {
                        Ok(())
                    }
                },
            };
            visitor.visit_value(place)?;
        }
        // The part between the end_ptr and the end of the place is also frozen.
        // So pretend there is a 0-sized `UnsafeCell` at the end.
        unsafe_cell_action(place.ptr.wrapping_offset(size, this), Size::ZERO)?;
        // Done!
        return Ok(());

        /// Visiting the memory covered by a `MemPlace`, being aware of
        /// whether we are inside an `UnsafeCell` or not.
        struct UnsafeCellVisitor<'ecx, 'mir, 'tcx, F>
        where
            F: FnMut(&MPlaceTy<'tcx, Tag>) -> InterpResult<'tcx>,
        {
            ecx: &'ecx MiriEvalContext<'mir, 'tcx>,
            unsafe_cell_action: F,
        }

        impl<'ecx, 'mir, 'tcx: 'mir, F> ValueVisitor<'mir, 'tcx, Evaluator<'mir, 'tcx>>
            for UnsafeCellVisitor<'ecx, 'mir, 'tcx, F>
        where
            F: FnMut(&MPlaceTy<'tcx, Tag>) -> InterpResult<'tcx>,
        {
            type V = MPlaceTy<'tcx, Tag>;

            #[inline(always)]
            fn ecx(&self) -> &MiriEvalContext<'mir, 'tcx> {
                &self.ecx
            }

            // Hook to detect `UnsafeCell`.
            fn visit_value(&mut self, v: &MPlaceTy<'tcx, Tag>) -> InterpResult<'tcx> {
                trace!("UnsafeCellVisitor: {:?} {:?}", *v, v.layout.ty);
                let is_unsafe_cell = match v.layout.ty.kind() {
                    ty::Adt(adt, _) =>
                        Some(adt.did()) == self.ecx.tcx.lang_items().unsafe_cell_type(),
                    _ => false,
                };
                if is_unsafe_cell {
                    // We do not have to recurse further, this is an `UnsafeCell`.
                    (self.unsafe_cell_action)(v)
                } else if self.ecx.type_is_freeze(v.layout.ty) {
                    // This is `Freeze`, there cannot be an `UnsafeCell`
                    Ok(())
                } else if matches!(v.layout.fields, FieldsShape::Union(..)) {
                    // A (non-frozen) union. We fall back to whatever the type says.
                    (self.unsafe_cell_action)(v)
                } else {
                    // We want to not actually read from memory for this visit. So, before
                    // walking this value, we have to make sure it is not a
                    // `Variants::Multiple`.
                    match v.layout.variants {
                        Variants::Multiple { .. } => {
                            // A multi-variant enum, or generator, or so.
                            // Treat this like a union: without reading from memory,
                            // we cannot determine the variant we are in. Reading from
                            // memory would be subject to Stacked Borrows rules, leading
                            // to all sorts of "funny" recursion.
                            // We only end up here if the type is *not* freeze, so we just call the
                            // `UnsafeCell` action.
                            (self.unsafe_cell_action)(v)
                        }
                        Variants::Single { .. } => {
                            // Proceed further, try to find where exactly that `UnsafeCell`
                            // is hiding.
                            self.walk_value(v)
                        }
                    }
                }
            }

            // Make sure we visit aggregrates in increasing offset order.
            fn visit_aggregate(
                &mut self,
                place: &MPlaceTy<'tcx, Tag>,
                fields: impl Iterator<Item = InterpResult<'tcx, MPlaceTy<'tcx, Tag>>>,
            ) -> InterpResult<'tcx> {
                match place.layout.fields {
                    FieldsShape::Array { .. } => {
                        // For the array layout, we know the iterator will yield sorted elements so
                        // we can avoid the allocation.
                        self.walk_aggregate(place, fields)
                    }
                    FieldsShape::Arbitrary { .. } => {
                        // Gather the subplaces and sort them before visiting.
                        let mut places =
                            fields.collect::<InterpResult<'tcx, Vec<MPlaceTy<'tcx, Tag>>>>()?;
                        // we just compare offsets, the abs. value never matters
                        places.sort_by_key(|place| {
                            place.ptr.into_pointer_or_addr().unwrap().into_parts().1 as Size
                        });
                        self.walk_aggregate(place, places.into_iter().map(Ok))
                    }
                    FieldsShape::Union { .. } | FieldsShape::Primitive => {
                        // Uh, what?
                        bug!("unions/primitives are not aggregates we should ever visit")
                    }
                }
            }

            fn visit_union(
                &mut self,
                _v: &MPlaceTy<'tcx, Tag>,
                _fields: NonZeroUsize,
            ) -> InterpResult<'tcx> {
                bug!("we should have already handled unions in `visit_value`")
            }
        }
    }

    /// Helper function used inside the shims of foreign functions to check that isolation is
    /// disabled. It returns an error using the `name` of the foreign function if this is not the
    /// case.
    fn check_no_isolation(&self, name: &str) -> InterpResult<'tcx> {
        if !self.eval_context_ref().machine.communicate() {
            self.reject_in_isolation(name, RejectOpWith::Abort)?;
        }
        Ok(())
    }

    /// Helper function used inside the shims of foreign functions which reject the op
    /// when isolation is enabled. It is used to print a warning/backtrace about the rejection.
    fn reject_in_isolation(&self, op_name: &str, reject_with: RejectOpWith) -> InterpResult<'tcx> {
        let this = self.eval_context_ref();
        match reject_with {
            RejectOpWith::Abort => isolation_abort_error(op_name),
            RejectOpWith::WarningWithoutBacktrace => {
                this.tcx
                    .sess
                    .warn(&format!("{} was made to return an error due to isolation", op_name));
                Ok(())
            }
            RejectOpWith::Warning => {
                register_diagnostic(NonHaltingDiagnostic::RejectedIsolatedOp(op_name.to_string()));
                Ok(())
            }
            RejectOpWith::NoWarning => Ok(()), // no warning
        }
    }

    /// Helper function used inside the shims of foreign functions to assert that the target OS
    /// is `target_os`. It panics showing a message with the `name` of the foreign function
    /// if this is not the case.
    fn assert_target_os(&self, target_os: &str, name: &str) {
        assert_eq!(
            self.eval_context_ref().tcx.sess.target.os,
            target_os,
            "`{}` is only available on the `{}` target OS",
            name,
            target_os,
        )
    }

    /// Get last error variable as a place, lazily allocating thread-local storage for it if
    /// necessary.
    fn last_error_place(&mut self) -> InterpResult<'tcx, MPlaceTy<'tcx, Tag>> {
        let this = self.eval_context_mut();
        if let Some(errno_place) = this.active_thread_ref().last_error {
            Ok(errno_place)
        } else {
            // Allocate new place, set initial value to 0.
            let errno_layout = this.machine.layouts.u32;
            let errno_place = this.allocate(errno_layout, MiriMemoryKind::Machine.into())?;
            this.write_scalar(Scalar::from_u32(0), &errno_place.into())?;
            this.active_thread_mut().last_error = Some(errno_place);
            Ok(errno_place)
        }
    }

    /// Sets the last error variable.
    fn set_last_error(&mut self, scalar: Scalar<Tag>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let errno_place = this.last_error_place()?;
        this.write_scalar(scalar, &errno_place.into())
    }

    /// Gets the last error variable.
    fn get_last_error(&mut self) -> InterpResult<'tcx, Scalar<Tag>> {
        let this = self.eval_context_mut();
        let errno_place = this.last_error_place()?;
        this.read_scalar(&errno_place.into())?.check_init()
    }

    /// Sets the last OS error using a `std::io::ErrorKind`. This function tries to produce the most
    /// similar OS error from the `std::io::ErrorKind` and sets it as the last OS error.
    fn set_last_error_from_io_error(&mut self, err_kind: std::io::ErrorKind) -> InterpResult<'tcx> {
        use std::io::ErrorKind::*;
        let this = self.eval_context_mut();
        let target = &this.tcx.sess.target;
        let target_os = &target.os;
        let last_error = if target.families.contains(&"unix".to_owned()) {
            this.eval_libc(match err_kind {
                ConnectionRefused => "ECONNREFUSED",
                ConnectionReset => "ECONNRESET",
                PermissionDenied => "EPERM",
                BrokenPipe => "EPIPE",
                NotConnected => "ENOTCONN",
                ConnectionAborted => "ECONNABORTED",
                AddrNotAvailable => "EADDRNOTAVAIL",
                AddrInUse => "EADDRINUSE",
                NotFound => "ENOENT",
                Interrupted => "EINTR",
                InvalidInput => "EINVAL",
                TimedOut => "ETIMEDOUT",
                AlreadyExists => "EEXIST",
                WouldBlock => "EWOULDBLOCK",
                DirectoryNotEmpty => "ENOTEMPTY",
                _ => {
                    throw_unsup_format!(
                        "io error {:?} cannot be translated into a raw os error",
                        err_kind
                    )
                }
            })?
        } else if target.families.contains(&"windows".to_owned()) {
            // FIXME: we have to finish implementing the Windows equivalent of this.
            this.eval_windows(
                "c",
                match err_kind {
                    NotFound => "ERROR_FILE_NOT_FOUND",
                    PermissionDenied => "ERROR_ACCESS_DENIED",
                    _ =>
                        throw_unsup_format!(
                            "io error {:?} cannot be translated into a raw os error",
                            err_kind
                        ),
                },
            )?
        } else {
            throw_unsup_format!(
                "setting the last OS error from an io::Error is unsupported for {}.",
                target_os
            )
        };
        this.set_last_error(last_error)
    }

    /// Helper function that consumes an `std::io::Result<T>` and returns an
    /// `InterpResult<'tcx,T>::Ok` instead. In case the result is an error, this function returns
    /// `Ok(-1)` and sets the last OS error accordingly.
    ///
    /// This function uses `T: From<i32>` instead of `i32` directly because some IO related
    /// functions return different integer types (like `read`, that returns an `i64`).
    fn try_unwrap_io_result<T: From<i32>>(
        &mut self,
        result: std::io::Result<T>,
    ) -> InterpResult<'tcx, T> {
        match result {
            Ok(ok) => Ok(ok),
            Err(e) => {
                self.eval_context_mut().set_last_error_from_io_error(e.kind())?;
                Ok((-1).into())
            }
        }
    }

    fn read_scalar_at_offset(
        &self,
        op: &OpTy<'tcx, Tag>,
        offset: u64,
        layout: TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx, ScalarMaybeUninit<Tag>> {
        let this = self.eval_context_ref();
        let op_place = this.deref_operand(op)?;
        let offset = Size::from_bytes(offset);
        // Ensure that the following read at an offset is within bounds
        assert!(op_place.layout.size >= offset + layout.size);
        let value_place = op_place.offset(offset, MemPlaceMeta::None, layout, this)?;
        this.read_scalar(&value_place.into())
    }

    fn write_scalar_at_offset(
        &mut self,
        op: &OpTy<'tcx, Tag>,
        offset: u64,
        value: impl Into<ScalarMaybeUninit<Tag>>,
        layout: TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();
        let op_place = this.deref_operand(op)?;
        let offset = Size::from_bytes(offset);
        // Ensure that the following read at an offset is within bounds
        assert!(op_place.layout.size >= offset + layout.size);
        let value_place = op_place.offset(offset, MemPlaceMeta::None, layout, this)?;
        this.write_scalar(value, &value_place.into())
    }

    /// Parse a `timespec` struct and return it as a `std::time::Duration`. It returns `None`
    /// if the value in the `timespec` struct is invalid. Some libc functions will return
    /// `EINVAL` in this case.
    fn read_timespec(&mut self, tp: &MPlaceTy<'tcx, Tag>) -> InterpResult<'tcx, Option<Duration>> {
        let this = self.eval_context_mut();
        let seconds_place = this.mplace_field(&tp, 0)?;
        let seconds_scalar = this.read_scalar(&seconds_place.into())?;
        let seconds = seconds_scalar.to_machine_isize(this)?;
        let nanoseconds_place = this.mplace_field(&tp, 1)?;
        let nanoseconds_scalar = this.read_scalar(&nanoseconds_place.into())?;
        let nanoseconds = nanoseconds_scalar.to_machine_isize(this)?;

        Ok(try {
            // tv_sec must be non-negative.
            let seconds: u64 = seconds.try_into().ok()?;
            // tv_nsec must be non-negative.
            let nanoseconds: u32 = nanoseconds.try_into().ok()?;
            if nanoseconds >= 1_000_000_000 {
                // tv_nsec must not be greater than 999,999,999.
                None?
            }
            Duration::new(seconds, nanoseconds)
        })
    }

    fn read_c_str<'a>(&'a self, ptr: Pointer<Option<Tag>>) -> InterpResult<'tcx, &'a [u8]>
    where
        'tcx: 'a,
        'mir: 'a,
    {
        let this = self.eval_context_ref();
        let size1 = Size::from_bytes(1);

        // Step 1: determine the length.
        let mut len = Size::ZERO;
        loop {
            // FIXME: We are re-getting the allocation each time around the loop.
            // Would be nice if we could somehow "extend" an existing AllocRange.
            let alloc = this.memory.get(ptr.offset(len, this)?.into(), size1, Align::ONE)?.unwrap(); // not a ZST, so we will get a result
            let byte = alloc.read_scalar(alloc_range(Size::ZERO, size1))?.to_u8()?;
            if byte == 0 {
                break;
            } else {
                len = len + size1;
            }
        }

        // Step 2: get the bytes.
        this.memory.read_bytes(ptr.into(), len)
    }

    fn read_wide_str(&self, mut ptr: Pointer<Option<Tag>>) -> InterpResult<'tcx, Vec<u16>> {
        let this = self.eval_context_ref();
        let size2 = Size::from_bytes(2);
        let align2 = Align::from_bytes(2).unwrap();

        let mut wchars = Vec::new();
        loop {
            // FIXME: We are re-getting the allocation each time around the loop.
            // Would be nice if we could somehow "extend" an existing AllocRange.
            let alloc = this.memory.get(ptr.into(), size2, align2)?.unwrap(); // not a ZST, so we will get a result
            let wchar = alloc.read_scalar(alloc_range(Size::ZERO, size2))?.to_u16()?;
            if wchar == 0 {
                break;
            } else {
                wchars.push(wchar);
                ptr = ptr.offset(size2, this)?;
            }
        }

        Ok(wchars)
    }

    /// Check that the ABI is what we expect.
    fn check_abi<'a>(&self, abi: Abi, exp_abi: Abi) -> InterpResult<'a, ()> {
        if self.eval_context_ref().machine.enforce_abi && abi != exp_abi {
            throw_ub_format!(
                "calling a function with ABI {} using caller ABI {}",
                exp_abi.name(),
                abi.name()
            )
        }
        Ok(())
    }

    fn frame_in_std(&self) -> bool {
        let this = self.eval_context_ref();
        this.tcx.lang_items().start_fn().map_or(false, |start_fn| {
            this.tcx.def_path(this.frame().instance.def_id()).krate
                == this.tcx.def_path(start_fn).krate
        })
    }

    /// Handler that should be called when unsupported functionality is encountered.
    /// This function will either panic within the context of the emulated application
    /// or return an error in the Miri process context
    ///
    /// Return value of `Ok(bool)` indicates whether execution should continue.
    fn handle_unsupported<S: AsRef<str>>(&mut self, error_msg: S) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();
        if this.machine.panic_on_unsupported {
            // message is slightly different here to make automated analysis easier
            let error_msg = format!("unsupported Miri functionality: {}", error_msg.as_ref());
            this.start_panic(error_msg.as_ref(), StackPopUnwind::Skip)?;
            return Ok(());
        } else {
            throw_unsup_format!("{}", error_msg.as_ref());
        }
    }

    fn check_abi_and_shim_symbol_clash(
        &mut self,
        abi: Abi,
        exp_abi: Abi,
        link_name: Symbol,
    ) -> InterpResult<'tcx, ()> {
        self.check_abi(abi, exp_abi)?;
        if let Some((body, _)) = self.eval_context_mut().lookup_exported_symbol(link_name)? {
            throw_machine_stop!(TerminationInfo::SymbolShimClashing {
                link_name,
                span: body.span.data(),
            })
        }
        Ok(())
    }

    fn check_shim<'a, const N: usize>(
        &mut self,
        abi: Abi,
        exp_abi: Abi,
        link_name: Symbol,
        args: &'a [OpTy<'tcx, Tag>],
    ) -> InterpResult<'tcx, &'a [OpTy<'tcx, Tag>; N]>
    where
        &'a [OpTy<'tcx, Tag>; N]: TryFrom<&'a [OpTy<'tcx, Tag>]>,
    {
        self.check_abi_and_shim_symbol_clash(abi, exp_abi, link_name)?;
        check_arg_count(args)
    }

    /// Mark a machine allocation that was just created as immutable.
    fn mark_immutable(&mut self, mplace: &MemPlace<Tag>) {
        let this = self.eval_context_mut();
        this.memory
            .mark_immutable(mplace.ptr.into_pointer_or_addr().unwrap().provenance.alloc_id)
            .unwrap();
    }
}

/// Check that the number of args is what we expect.
pub fn check_arg_count<'a, 'tcx, const N: usize>(
    args: &'a [OpTy<'tcx, Tag>],
) -> InterpResult<'tcx, &'a [OpTy<'tcx, Tag>; N]>
where
    &'a [OpTy<'tcx, Tag>; N]: TryFrom<&'a [OpTy<'tcx, Tag>]>,
{
    if let Ok(ops) = args.try_into() {
        return Ok(ops);
    }
    throw_ub_format!("incorrect number of arguments: got {}, expected {}", args.len(), N)
}

pub fn isolation_abort_error(name: &str) -> InterpResult<'static> {
    throw_machine_stop!(TerminationInfo::UnsupportedInIsolation(format!(
        "{} not available when isolation is enabled",
        name,
    )))
}
