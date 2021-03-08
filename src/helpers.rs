use std::convert::{TryFrom, TryInto};
use std::mem;
use std::num::NonZeroUsize;
use std::time::Duration;

use log::trace;

use rustc_middle::mir;
use rustc_middle::ty::{self, List, TyCtxt, layout::TyAndLayout};
use rustc_hir::def_id::{DefId, CRATE_DEF_INDEX};
use rustc_target::abi::{LayoutOf, Size, FieldsShape, Variants};
use rustc_target::spec::abi::Abi;

use rand::RngCore;

use crate::*;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}

/// Gets an instance for a path.
fn try_resolve_did<'mir, 'tcx>(tcx: TyCtxt<'tcx>, path: &[&str]) -> Option<DefId> {
    tcx.crates()
        .iter()
        .find(|&&krate| tcx.original_crate_name(krate).as_str() == path[0])
        .and_then(|krate| {
            let krate = DefId { krate: *krate, index: CRATE_DEF_INDEX };
            let mut items = tcx.item_children(krate);
            let mut path_it = path.iter().skip(1).peekable();

            while let Some(segment) = path_it.next() {
                for item in mem::replace(&mut items, Default::default()).iter() {
                    if item.ident.name.as_str() == *segment {
                        if path_it.peek().is_none() {
                            return Some(item.res.def_id());
                        }

                        items = tcx.item_children(item.res.def_id());
                        break;
                    }
                }
            }
            None
        })
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
    fn eval_path_scalar(
        &mut self,
        path: &[&str],
    ) -> InterpResult<'tcx, ScalarMaybeUninit<Tag>> {
        let this = self.eval_context_mut();
        let instance = this.resolve_path(path);
        let cid = GlobalId { instance, promoted: None };
        let const_val = this.eval_to_allocation(cid)?;
        let const_val = this.read_scalar(&const_val.into())?;
        return Ok(const_val);
    }

    /// Helper function to get a `libc` constant as a `Scalar`.
    fn eval_libc(&mut self, name: &str) -> InterpResult<'tcx, Scalar<Tag>> {
        self.eval_context_mut()
            .eval_path_scalar(&["libc", name])?
            .check_init()
    }

    /// Helper function to get a `libc` constant as an `i32`.
    fn eval_libc_i32(&mut self, name: &str) -> InterpResult<'tcx, i32> {
        // TODO: Cache the result.
        self.eval_libc(name)?.to_i32()
    }

    /// Helper function to get a `windows` constant as a `Scalar`.
    fn eval_windows(&mut self, module: &str, name: &str) -> InterpResult<'tcx, Scalar<Tag>> {
        self.eval_context_mut()
            .eval_path_scalar(&["std", "sys", "windows", module, name])?
            .check_init()
    }

    /// Helper function to get a `windows` constant as an `u64`.
    fn eval_windows_u64(&mut self, module: &str, name: &str) -> InterpResult<'tcx, u64> {
        // TODO: Cache the result.
        self.eval_windows(module, name)?.to_u64()
    }

    /// Helper function to get the `TyAndLayout` of a `libc` type
    fn libc_ty_layout(&mut self, name: &str) -> InterpResult<'tcx, TyAndLayout<'tcx>> {
        let this = self.eval_context_mut();
        let ty = this.resolve_path(&["libc", name]).ty(*this.tcx, ty::ParamEnv::reveal_all());
        this.layout_of(ty)
    }

    /// Helper function to get the `TyAndLayout` of a `windows` type
    fn windows_ty_layout(&mut self, name: &str) -> InterpResult<'tcx, TyAndLayout<'tcx>> {
        let this = self.eval_context_mut();
        let ty = this.resolve_path(&["std", "sys", "windows", "c", name]).ty(*this.tcx, ty::ParamEnv::reveal_all());
        this.layout_of(ty)
    }

    /// Write a 0 of the appropriate size to `dest`.
    fn write_null(&mut self, dest: &PlaceTy<'tcx, Tag>) -> InterpResult<'tcx> {
        self.eval_context_mut().write_scalar(Scalar::from_int(0, dest.layout.size), dest)
    }

    /// Test if this immediate equals 0.
    fn is_null(&self, val: Scalar<Tag>) -> InterpResult<'tcx, bool> {
        let this = self.eval_context_ref();
        let null = Scalar::null_ptr(this);
        this.ptr_eq(val, null)
    }

    /// Turn a Scalar into an Option<NonNullScalar>
    fn test_null(&self, val: Scalar<Tag>) -> InterpResult<'tcx, Option<Scalar<Tag>>> {
        let this = self.eval_context_ref();
        Ok(if this.is_null(val)? { None } else { Some(val) })
    }

    /// Get the `Place` for a local
    fn local_place(&mut self, local: mir::Local) -> InterpResult<'tcx, PlaceTy<'tcx, Tag>> {
        let this = self.eval_context_mut();
        let place = mir::Place { local: local, projection: List::empty() };
        this.eval_place(place)
    }

    /// Generate some random bytes, and write them to `dest`.
    fn gen_random(&mut self, ptr: Scalar<Tag>, len: u64) -> InterpResult<'tcx> {
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

        if this.machine.communicate {
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
        args: &[Immediate<Tag>],
        dest: Option<&PlaceTy<'tcx, Tag>>,
        stack_pop: StackPopCleanup,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        // Push frame.
        let mir = &*this.load_mir(f.def, None)?;
        this.push_stack_frame(f, mir, dest, stack_pop)?;

        // Initialize arguments.
        let mut callee_args = this.frame().body.args_iter();
        for arg in args {
            let callee_arg = this.local_place(
                callee_args.next().expect("callee has fewer arguments than expected"),
            )?;
            this.write_immediate(*arg, &callee_arg)?;
        }
        assert_eq!(callee_args.next(), None, "callee has more arguments than expected");

        Ok(())
    }

    /// Visits the memory covered by `place`, sensitive to freezing: the 3rd parameter
    /// will be true if this is frozen, false if this is in an `UnsafeCell`.
    fn visit_freeze_sensitive(
        &self,
        place: &MPlaceTy<'tcx, Tag>,
        size: Size,
        mut action: impl FnMut(Pointer<Tag>, Size, bool) -> InterpResult<'tcx>,
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
        let mut end_ptr = place.ptr.assert_ptr();
        // Called when we detected an `UnsafeCell` at the given offset and size.
        // Calls `action` and advances `end_ptr`.
        let mut unsafe_cell_action = |unsafe_cell_ptr: Scalar<Tag>, unsafe_cell_size: Size| {
            let unsafe_cell_ptr = unsafe_cell_ptr.assert_ptr();
            debug_assert_eq!(unsafe_cell_ptr.alloc_id, end_ptr.alloc_id);
            debug_assert_eq!(unsafe_cell_ptr.tag, end_ptr.tag);
            // We assume that we are given the fields in increasing offset order,
            // and nothing else changes.
            let unsafe_cell_offset = unsafe_cell_ptr.offset;
            let end_offset = end_ptr.offset;
            assert!(unsafe_cell_offset >= end_offset);
            let frozen_size = unsafe_cell_offset - end_offset;
            // Everything between the end_ptr and this `UnsafeCell` is frozen.
            if frozen_size != Size::ZERO {
                action(end_ptr, frozen_size, /*frozen*/ true)?;
            }
            // This `UnsafeCell` is NOT frozen.
            if unsafe_cell_size != Size::ZERO {
                action(unsafe_cell_ptr, unsafe_cell_size, /*frozen*/ false)?;
            }
            // Update end end_ptr.
            end_ptr = unsafe_cell_ptr.wrapping_offset(unsafe_cell_size, this);
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
        unsafe_cell_action(place.ptr.ptr_wrapping_offset(size, this), Size::ZERO)?;
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
                        Some(adt.did) == self.ecx.tcx.lang_items().unsafe_cell_type(),
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
                        places.sort_by_key(|place| place.ptr.assert_ptr().offset);
                        self.walk_aggregate(place, places.into_iter().map(Ok))
                    }
                    FieldsShape::Union { .. } | FieldsShape::Primitive => {
                        // Uh, what?
                        bug!("unions/primitives are not aggregates we should ever visit")
                    }
                }
            }

            fn visit_union(&mut self, _v: &MPlaceTy<'tcx, Tag>, _fields: NonZeroUsize) -> InterpResult<'tcx> {
                bug!("we should have already handled unions in `visit_value`")
            }
        }
    }

    // Writes several `ImmTy`s contiguously into memory. This is useful when you have to pack
    // different values into a struct.
    fn write_packed_immediates(
        &mut self,
        place: &MPlaceTy<'tcx, Tag>,
        imms: &[ImmTy<'tcx, Tag>],
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let mut offset = Size::from_bytes(0);

        for &imm in imms {
            this.write_immediate_to_mplace(
                *imm,
                &place.offset(offset, MemPlaceMeta::None, imm.layout, &*this.tcx)?,
            )?;
            offset += imm.layout.size;
        }
        Ok(())
    }

    /// Helper function used inside the shims of foreign functions to check that isolation is
    /// disabled. It returns an error using the `name` of the foreign function if this is not the
    /// case.
    fn check_no_isolation(&self, name: &str) -> InterpResult<'tcx> {
        if !self.eval_context_ref().machine.communicate {
            isolation_error(name)?;
        }
        Ok(())
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
            let errno_place = this.allocate(errno_layout, MiriMemoryKind::Machine.into());
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

    /// Sets the last OS error using a `std::io::Error`. This function tries to produce the most
    /// similar OS error from the `std::io::ErrorKind` and sets it as the last OS error.
    fn set_last_error_from_io_error(&mut self, e: std::io::Error) -> InterpResult<'tcx> {
        use std::io::ErrorKind::*;
        let this = self.eval_context_mut();
        let target = &this.tcx.sess.target;
        let target_os = &target.os;
        let last_error = if target.os_family == Some("unix".to_owned()) {
            this.eval_libc(match e.kind() {
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
                _ => {
                    throw_unsup_format!("io error {} cannot be transformed into a raw os error", e)
                }
            })?
        } else if target_os == "windows" {
            // FIXME: we have to finish implementing the Windows equivalent of this.
            this.eval_windows("c", match e.kind() {
                NotFound => "ERROR_FILE_NOT_FOUND",
                _ => throw_unsup_format!("io error {} cannot be transformed into a raw os error", e)
            })?
        } else {
            throw_unsup_format!("setting the last OS error from an io::Error is unsupported for {}.", target_os)
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
                self.eval_context_mut().set_last_error_from_io_error(e)?;
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
    fn read_timespec(
        &mut self,
        timespec_ptr_op: &OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, Option<Duration>> {
        let this = self.eval_context_mut();
        let tp = this.deref_operand(timespec_ptr_op)?;
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
}

/// Check that the number of args is what we expect.
pub fn check_arg_count<'a, 'tcx, const N: usize>(args: &'a [OpTy<'tcx, Tag>]) -> InterpResult<'tcx, &'a [OpTy<'tcx, Tag>; N]>
    where &'a [OpTy<'tcx, Tag>; N]: TryFrom<&'a [OpTy<'tcx, Tag>]> {
    if let Ok(ops) = args.try_into() {
        return Ok(ops);
    }
    throw_ub_format!("incorrect number of arguments: got {}, expected {}", args.len(), N)
}

/// Check that the ABI is what we expect.
pub fn check_abi<'a>(abi: Abi, exp_abi: Abi) -> InterpResult<'a, ()> {
    if abi == exp_abi {
        Ok(())
    } else {
        throw_ub_format!("calling a function with ABI {:?} using caller ABI {:?}", exp_abi, abi)
    }
}

pub fn isolation_error(name: &str) -> InterpResult<'static> {
    throw_machine_stop!(TerminationInfo::UnsupportedInIsolation(format!(
        "{} not available when isolation is enabled",
        name,
    )))
}

pub fn immty_from_int_checked<'tcx>(
    int: impl Into<i128>,
    layout: TyAndLayout<'tcx>,
) -> InterpResult<'tcx, ImmTy<'tcx, Tag>> {
    let int = int.into();
    Ok(ImmTy::try_from_int(int, layout).ok_or_else(|| {
        err_unsup_format!("signed value {:#x} does not fit in {} bits", int, layout.size.bits())
    })?)
}

pub fn immty_from_uint_checked<'tcx>(
    int: impl Into<u128>,
    layout: TyAndLayout<'tcx>,
) -> InterpResult<'tcx, ImmTy<'tcx, Tag>> {
    let int = int.into();
    Ok(ImmTy::try_from_uint(int, layout).ok_or_else(|| {
        err_unsup_format!("unsigned value {:#x} does not fit in {} bits", int, layout.size.bits())
    })?)
}
