pub mod convert;

use std::any::Any;
use std::cmp;
use std::iter;
use std::num::NonZeroUsize;
use std::time::Duration;

use log::trace;

use rustc_hir::def::{DefKind, Namespace};
use rustc_hir::def_id::{DefId, CRATE_DEF_INDEX};
use rustc_middle::mir;
use rustc_middle::ty::{
    self,
    layout::{LayoutOf, TyAndLayout},
    List, TyCtxt,
};
use rustc_span::{def_id::CrateNum, sym, Span, Symbol};
use rustc_target::abi::{Align, FieldsShape, Size, Variants};
use rustc_target::spec::abi::Abi;

use rand::RngCore;

use crate::*;

/// A trait to work around not having trait object upcasting:
/// Add `AsAny` as supertrait and your trait objects can be turned into `&dyn Any` on which you can
/// then call `downcast`.
pub trait AsAny: Any {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}
impl<T: Any> AsAny for T {
    #[inline(always)]
    fn as_any(&self) -> &dyn Any {
        self
    }
    #[inline(always)]
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// This mapping should match `decode_error_kind` in
// <https://github.com/rust-lang/rust/blob/master/library/std/src/sys/unix/mod.rs>.
const UNIX_IO_ERROR_TABLE: &[(&str, std::io::ErrorKind)] = {
    use std::io::ErrorKind::*;
    &[
        ("E2BIG", ArgumentListTooLong),
        ("EADDRINUSE", AddrInUse),
        ("EADDRNOTAVAIL", AddrNotAvailable),
        ("EBUSY", ResourceBusy),
        ("ECONNABORTED", ConnectionAborted),
        ("ECONNREFUSED", ConnectionRefused),
        ("ECONNRESET", ConnectionReset),
        ("EDEADLK", Deadlock),
        ("EDQUOT", FilesystemQuotaExceeded),
        ("EEXIST", AlreadyExists),
        ("EFBIG", FileTooLarge),
        ("EHOSTUNREACH", HostUnreachable),
        ("EINTR", Interrupted),
        ("EINVAL", InvalidInput),
        ("EISDIR", IsADirectory),
        ("ELOOP", FilesystemLoop),
        ("ENOENT", NotFound),
        ("ENOMEM", OutOfMemory),
        ("ENOSPC", StorageFull),
        ("ENOSYS", Unsupported),
        ("EMLINK", TooManyLinks),
        ("ENAMETOOLONG", InvalidFilename),
        ("ENETDOWN", NetworkDown),
        ("ENETUNREACH", NetworkUnreachable),
        ("ENOTCONN", NotConnected),
        ("ENOTDIR", NotADirectory),
        ("ENOTEMPTY", DirectoryNotEmpty),
        ("EPIPE", BrokenPipe),
        ("EROFS", ReadOnlyFilesystem),
        ("ESPIPE", NotSeekable),
        ("ESTALE", StaleNetworkFileHandle),
        ("ETIMEDOUT", TimedOut),
        ("ETXTBSY", ExecutableFileBusy),
        ("EXDEV", CrossesDevices),
        // The following have two valid options. We have both for the forwards mapping; only the
        // first one will be used for the backwards mapping.
        ("EPERM", PermissionDenied),
        ("EACCES", PermissionDenied),
        ("EWOULDBLOCK", WouldBlock),
        ("EAGAIN", WouldBlock),
    ]
};

/// Gets an instance for a path.
///
/// A `None` namespace indicates we are looking for a module.
fn try_resolve_did(tcx: TyCtxt<'_>, path: &[&str], namespace: Option<Namespace>) -> Option<DefId> {
    /// Yield all children of the given item, that have the given name.
    fn find_children<'tcx: 'a, 'a>(
        tcx: TyCtxt<'tcx>,
        item: DefId,
        name: &'a str,
    ) -> impl Iterator<Item = DefId> + 'a {
        tcx.module_children(item)
            .iter()
            .filter(move |item| item.ident.name.as_str() == name)
            .map(move |item| item.res.def_id())
    }

    // Take apart the path: leading crate, a sequence of modules, and potentially a final item.
    let (&crate_name, path) = path.split_first().expect("paths must have at least one segment");
    let (modules, item) = if let Some(namespace) = namespace {
        let (&item_name, modules) =
            path.split_last().expect("non-module paths must have at least 2 segments");
        (modules, Some((item_name, namespace)))
    } else {
        (path, None)
    };

    // First find the crate.
    let krate =
        tcx.crates(()).iter().find(|&&krate| tcx.crate_name(krate).as_str() == crate_name)?;
    let mut cur_item = DefId { krate: *krate, index: CRATE_DEF_INDEX };
    // Then go over the modules.
    for &segment in modules {
        cur_item = find_children(tcx, cur_item, segment)
            .find(|item| tcx.def_kind(item) == DefKind::Mod)?;
    }
    // Finally, look up the desired item in this module, if any.
    match item {
        Some((item_name, namespace)) =>
            Some(
                find_children(tcx, cur_item, item_name)
                    .find(|item| tcx.def_kind(item).ns() == Some(namespace))?,
            ),
        None => Some(cur_item),
    }
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    /// Checks if the given crate/module exists.
    fn have_module(&self, path: &[&str]) -> bool {
        try_resolve_did(*self.eval_context_ref().tcx, path, None).is_some()
    }

    /// Gets an instance for a path; fails gracefully if the path does not exist.
    fn try_resolve_path(&self, path: &[&str], namespace: Namespace) -> Option<ty::Instance<'tcx>> {
        let tcx = self.eval_context_ref().tcx.tcx;
        let did = try_resolve_did(tcx, path, Some(namespace))?;
        Some(ty::Instance::mono(tcx, did))
    }

    /// Gets an instance for a path.
    fn resolve_path(&self, path: &[&str], namespace: Namespace) -> ty::Instance<'tcx> {
        self.try_resolve_path(path, namespace)
            .unwrap_or_else(|| panic!("failed to find required Rust item: {path:?}"))
    }

    /// Evaluates the scalar at the specified path.
    fn eval_path_scalar(&self, path: &[&str]) -> Scalar<Provenance> {
        let this = self.eval_context_ref();
        let instance = this.resolve_path(path, Namespace::ValueNS);
        let cid = GlobalId { instance, promoted: None };
        // We don't give a span -- this isn't actually used directly by the program anyway.
        let const_val = this
            .eval_global(cid, None)
            .unwrap_or_else(|err| panic!("failed to evaluate required Rust item: {path:?}\n{err}"));
        this.read_scalar(&const_val.into())
            .unwrap_or_else(|err| panic!("failed to read required Rust item: {path:?}\n{err}"))
    }

    /// Helper function to get a `libc` constant as a `Scalar`.
    fn eval_libc(&self, name: &str) -> Scalar<Provenance> {
        self.eval_path_scalar(&["libc", name])
    }

    /// Helper function to get a `libc` constant as an `i32`.
    fn eval_libc_i32(&self, name: &str) -> i32 {
        // TODO: Cache the result.
        self.eval_libc(name).to_i32().unwrap_or_else(|_err| {
            panic!("required libc item has unexpected type (not `i32`): {name}")
        })
    }

    /// Helper function to get a `libc` constant as an `u32`.
    fn eval_libc_u32(&self, name: &str) -> u32 {
        // TODO: Cache the result.
        self.eval_libc(name).to_u32().unwrap_or_else(|_err| {
            panic!("required libc item has unexpected type (not `u32`): {name}")
        })
    }

    /// Helper function to get a `windows` constant as a `Scalar`.
    fn eval_windows(&self, module: &str, name: &str) -> Scalar<Provenance> {
        self.eval_context_ref().eval_path_scalar(&["std", "sys", "windows", module, name])
    }

    /// Helper function to get a `windows` constant as a `u32`.
    fn eval_windows_u32(&self, module: &str, name: &str) -> u32 {
        // TODO: Cache the result.
        self.eval_windows(module, name).to_u32().unwrap_or_else(|_err| {
            panic!("required Windows item has unexpected type (not `u32`): {module}::{name}")
        })
    }

    /// Helper function to get a `windows` constant as a `u64`.
    fn eval_windows_u64(&self, module: &str, name: &str) -> u64 {
        // TODO: Cache the result.
        self.eval_windows(module, name).to_u64().unwrap_or_else(|_err| {
            panic!("required Windows item has unexpected type (not `u64`): {module}::{name}")
        })
    }

    /// Helper function to get the `TyAndLayout` of a `libc` type
    fn libc_ty_layout(&self, name: &str) -> TyAndLayout<'tcx> {
        let this = self.eval_context_ref();
        let ty = this
            .resolve_path(&["libc", name], Namespace::TypeNS)
            .ty(*this.tcx, ty::ParamEnv::reveal_all());
        this.layout_of(ty).unwrap()
    }

    /// Helper function to get the `TyAndLayout` of a `windows` type
    fn windows_ty_layout(&self, name: &str) -> TyAndLayout<'tcx> {
        let this = self.eval_context_ref();
        let ty = this
            .resolve_path(&["std", "sys", "windows", "c", name], Namespace::TypeNS)
            .ty(*this.tcx, ty::ParamEnv::reveal_all());
        this.layout_of(ty).unwrap()
    }

    /// Project to the given *named* field of the mplace (which must be a struct or union type).
    fn mplace_field_named(
        &self,
        mplace: &MPlaceTy<'tcx, Provenance>,
        name: &str,
    ) -> InterpResult<'tcx, MPlaceTy<'tcx, Provenance>> {
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
    fn write_int(
        &mut self,
        i: impl Into<i128>,
        dest: &PlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx> {
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
        dest: &MPlaceTy<'tcx, Provenance>,
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
        dest: &MPlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        for &(name, val) in values.iter() {
            let field = this.mplace_field_named(dest, name)?;
            this.write_int(val, &field.into())?;
        }
        Ok(())
    }

    /// Write a 0 of the appropriate size to `dest`.
    fn write_null(&mut self, dest: &PlaceTy<'tcx, Provenance>) -> InterpResult<'tcx> {
        self.write_int(0, dest)
    }

    /// Test if this pointer equals 0.
    fn ptr_is_null(&self, ptr: Pointer<Option<Provenance>>) -> InterpResult<'tcx, bool> {
        Ok(ptr.addr().bytes() == 0)
    }

    /// Get the `Place` for a local
    fn local_place(&mut self, local: mir::Local) -> InterpResult<'tcx, PlaceTy<'tcx, Provenance>> {
        let this = self.eval_context_mut();
        let place = mir::Place { local, projection: List::empty() };
        this.eval_place(place)
    }

    /// Generate some random bytes, and write them to `dest`.
    fn gen_random(&mut self, ptr: Pointer<Option<Provenance>>, len: u64) -> InterpResult<'tcx> {
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
            let rng = this.machine.rng.get_mut();
            rng.fill_bytes(&mut data);
        }

        this.write_bytes_ptr(ptr, data.iter().copied())
    }

    /// Call a function: Push the stack frame and pass the arguments.
    /// For now, arguments must be scalars (so that the caller does not have to know the layout).
    ///
    /// If you do not provie a return place, a dangling zero-sized place will be created
    /// for your convenience.
    fn call_function(
        &mut self,
        f: ty::Instance<'tcx>,
        caller_abi: Abi,
        args: &[Immediate<Provenance>],
        dest: Option<&PlaceTy<'tcx, Provenance>>,
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
        let mir = this.load_mir(f.def, None)?;
        let dest = match dest {
            Some(dest) => dest.clone(),
            None => MPlaceTy::fake_alloc_zst(this.layout_of(mir.return_ty())?).into(),
        };
        this.push_stack_frame(f, mir, &dest, stack_pop)?;

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
    fn visit_freeze_sensitive(
        &self,
        place: &MPlaceTy<'tcx, Provenance>,
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
        let start_addr = place.ptr.addr();
        let mut cur_addr = start_addr;
        // Called when we detected an `UnsafeCell` at the given offset and size.
        // Calls `action` and advances `cur_ptr`.
        let mut unsafe_cell_action = |unsafe_cell_ptr: &Pointer<Option<Provenance>>,
                                      unsafe_cell_size: Size| {
            // We assume that we are given the fields in increasing offset order,
            // and nothing else changes.
            let unsafe_cell_addr = unsafe_cell_ptr.addr();
            assert!(unsafe_cell_addr >= cur_addr);
            let frozen_size = unsafe_cell_addr - cur_addr;
            // Everything between the cur_ptr and this `UnsafeCell` is frozen.
            if frozen_size != Size::ZERO {
                action(alloc_range(cur_addr - start_addr, frozen_size), /*frozen*/ true)?;
            }
            cur_addr += frozen_size;
            // This `UnsafeCell` is NOT frozen.
            if unsafe_cell_size != Size::ZERO {
                action(
                    alloc_range(cur_addr - start_addr, unsafe_cell_size),
                    /*frozen*/ false,
                )?;
            }
            cur_addr += unsafe_cell_size;
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
                        .size_and_align_of_mplace(place)?
                        .map(|(size, _)| size)
                        // for extern types, just cover what we can
                        .unwrap_or_else(|| place.layout.size);
                    // Now handle this `UnsafeCell`, unless it is empty.
                    if unsafe_cell_size != Size::ZERO {
                        unsafe_cell_action(&place.ptr, unsafe_cell_size)
                    } else {
                        Ok(())
                    }
                },
            };
            visitor.visit_value(place)?;
        }
        // The part between the end_ptr and the end of the place is also frozen.
        // So pretend there is a 0-sized `UnsafeCell` at the end.
        unsafe_cell_action(&place.ptr.offset(size, this)?, Size::ZERO)?;
        // Done!
        return Ok(());

        /// Visiting the memory covered by a `MemPlace`, being aware of
        /// whether we are inside an `UnsafeCell` or not.
        struct UnsafeCellVisitor<'ecx, 'mir, 'tcx, F>
        where
            F: FnMut(&MPlaceTy<'tcx, Provenance>) -> InterpResult<'tcx>,
        {
            ecx: &'ecx MiriInterpCx<'mir, 'tcx>,
            unsafe_cell_action: F,
        }

        impl<'ecx, 'mir, 'tcx: 'mir, F> ValueVisitor<'mir, 'tcx, MiriMachine<'mir, 'tcx>>
            for UnsafeCellVisitor<'ecx, 'mir, 'tcx, F>
        where
            F: FnMut(&MPlaceTy<'tcx, Provenance>) -> InterpResult<'tcx>,
        {
            type V = MPlaceTy<'tcx, Provenance>;

            #[inline(always)]
            fn ecx(&self) -> &MiriInterpCx<'mir, 'tcx> {
                self.ecx
            }

            // Hook to detect `UnsafeCell`.
            fn visit_value(&mut self, v: &MPlaceTy<'tcx, Provenance>) -> InterpResult<'tcx> {
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
                } else if matches!(v.layout.ty.kind(), ty::Dynamic(_, _, ty::DynStar)) {
                    // This needs to read the vtable pointer to proceed type-driven, but we don't
                    // want to reentrantly read from memory here.
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
                place: &MPlaceTy<'tcx, Provenance>,
                fields: impl Iterator<Item = InterpResult<'tcx, MPlaceTy<'tcx, Provenance>>>,
            ) -> InterpResult<'tcx> {
                match place.layout.fields {
                    FieldsShape::Array { .. } => {
                        // For the array layout, we know the iterator will yield sorted elements so
                        // we can avoid the allocation.
                        self.walk_aggregate(place, fields)
                    }
                    FieldsShape::Arbitrary { .. } => {
                        // Gather the subplaces and sort them before visiting.
                        let mut places = fields
                            .collect::<InterpResult<'tcx, Vec<MPlaceTy<'tcx, Provenance>>>>()?;
                        // we just compare offsets, the abs. value never matters
                        places.sort_by_key(|place| place.ptr.addr());
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
                _v: &MPlaceTy<'tcx, Provenance>,
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
                    .warn(format!("{op_name} was made to return an error due to isolation"));
                Ok(())
            }
            RejectOpWith::Warning => {
                this.emit_diagnostic(NonHaltingDiagnostic::RejectedIsolatedOp(op_name.to_string()));
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
            "`{name}` is only available on the `{target_os}` target OS",
        )
    }

    /// Helper function used inside the shims of foreign functions to assert that the target OS
    /// is part of the UNIX family. It panics showing a message with the `name` of the foreign function
    /// if this is not the case.
    fn assert_target_os_is_unix(&self, name: &str) {
        assert!(
            target_os_is_unix(self.eval_context_ref().tcx.sess.target.os.as_ref()),
            "`{name}` is only available for supported UNIX family targets",
        );
    }

    /// Get last error variable as a place, lazily allocating thread-local storage for it if
    /// necessary.
    fn last_error_place(&mut self) -> InterpResult<'tcx, MPlaceTy<'tcx, Provenance>> {
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
    fn set_last_error(&mut self, scalar: Scalar<Provenance>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let errno_place = this.last_error_place()?;
        this.write_scalar(scalar, &errno_place.into())
    }

    /// Gets the last error variable.
    fn get_last_error(&mut self) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_mut();
        let errno_place = this.last_error_place()?;
        this.read_scalar(&errno_place.into())
    }

    /// This function tries to produce the most similar OS error from the `std::io::ErrorKind`
    /// as a platform-specific errnum.
    fn io_error_to_errnum(
        &self,
        err_kind: std::io::ErrorKind,
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_ref();
        let target = &this.tcx.sess.target;
        if target.families.iter().any(|f| f == "unix") {
            for &(name, kind) in UNIX_IO_ERROR_TABLE {
                if err_kind == kind {
                    return Ok(this.eval_libc(name));
                }
            }
            throw_unsup_format!("io error {:?} cannot be translated into a raw os error", err_kind)
        } else if target.families.iter().any(|f| f == "windows") {
            // FIXME: we have to finish implementing the Windows equivalent of this.
            use std::io::ErrorKind::*;
            Ok(this.eval_windows(
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
            ))
        } else {
            throw_unsup_format!(
                "converting io::Error into errnum is unsupported for OS {}",
                target.os
            )
        }
    }

    /// The inverse of `io_error_to_errnum`.
    #[allow(clippy::needless_return)]
    fn try_errnum_to_io_error(
        &self,
        errnum: Scalar<Provenance>,
    ) -> InterpResult<'tcx, Option<std::io::ErrorKind>> {
        let this = self.eval_context_ref();
        let target = &this.tcx.sess.target;
        if target.families.iter().any(|f| f == "unix") {
            let errnum = errnum.to_i32()?;
            for &(name, kind) in UNIX_IO_ERROR_TABLE {
                if errnum == this.eval_libc_i32(name) {
                    return Ok(Some(kind));
                }
            }
            // Our table is as complete as the mapping in std, so we are okay with saying "that's a
            // strange one" here.
            return Ok(None);
        } else {
            throw_unsup_format!(
                "converting errnum into io::Error is unsupported for OS {}",
                target.os
            )
        }
    }

    /// Sets the last OS error using a `std::io::ErrorKind`.
    fn set_last_error_from_io_error(&mut self, err_kind: std::io::ErrorKind) -> InterpResult<'tcx> {
        self.set_last_error(self.io_error_to_errnum(err_kind)?)
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

    /// Calculates the MPlaceTy given the offset and layout of an access on an operand
    fn deref_operand_and_offset(
        &self,
        op: &OpTy<'tcx, Provenance>,
        offset: u64,
        layout: TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx, MPlaceTy<'tcx, Provenance>> {
        let this = self.eval_context_ref();
        let op_place = this.deref_operand(op)?; // FIXME: we still deref with the original type!
        let offset = Size::from_bytes(offset);

        // Ensure that the access is within bounds.
        assert!(op_place.layout.size >= offset + layout.size);
        let value_place = op_place.offset(offset, layout, this)?;
        Ok(value_place)
    }

    fn read_scalar_at_offset(
        &self,
        op: &OpTy<'tcx, Provenance>,
        offset: u64,
        layout: TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_ref();
        let value_place = this.deref_operand_and_offset(op, offset, layout)?;
        this.read_scalar(&value_place.into())
    }

    fn write_scalar_at_offset(
        &mut self,
        op: &OpTy<'tcx, Provenance>,
        offset: u64,
        value: impl Into<Scalar<Provenance>>,
        layout: TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();
        let value_place = this.deref_operand_and_offset(op, offset, layout)?;
        this.write_scalar(value, &value_place.into())
    }

    /// Parse a `timespec` struct and return it as a `std::time::Duration`. It returns `None`
    /// if the value in the `timespec` struct is invalid. Some libc functions will return
    /// `EINVAL` in this case.
    fn read_timespec(
        &mut self,
        tp: &MPlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, Option<Duration>> {
        let this = self.eval_context_mut();
        let seconds_place = this.mplace_field(tp, 0)?;
        let seconds_scalar = this.read_scalar(&seconds_place.into())?;
        let seconds = seconds_scalar.to_target_isize(this)?;
        let nanoseconds_place = this.mplace_field(tp, 1)?;
        let nanoseconds_scalar = this.read_scalar(&nanoseconds_place.into())?;
        let nanoseconds = nanoseconds_scalar.to_target_isize(this)?;

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

    /// Read a sequence of bytes until the first null terminator.
    fn read_c_str<'a>(&'a self, ptr: Pointer<Option<Provenance>>) -> InterpResult<'tcx, &'a [u8]>
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
            let alloc = this.get_ptr_alloc(ptr.offset(len, this)?, size1, Align::ONE)?.unwrap(); // not a ZST, so we will get a result
            let byte = alloc.read_integer(alloc_range(Size::ZERO, size1))?.to_u8()?;
            if byte == 0 {
                break;
            } else {
                len += size1;
            }
        }

        // Step 2: get the bytes.
        this.read_bytes_ptr_strip_provenance(ptr, len)
    }

    /// Helper function to write a sequence of bytes with an added null-terminator, which is what
    /// the Unix APIs usually handle. This function returns `Ok((false, length))` without trying
    /// to write if `size` is not large enough to fit the contents of `c_str` plus a null
    /// terminator. It returns `Ok((true, length))` if the writing process was successful. The
    /// string length returned does include the null terminator.
    fn write_c_str(
        &mut self,
        c_str: &[u8],
        ptr: Pointer<Option<Provenance>>,
        size: u64,
    ) -> InterpResult<'tcx, (bool, u64)> {
        // If `size` is smaller or equal than `bytes.len()`, writing `bytes` plus the required null
        // terminator to memory using the `ptr` pointer would cause an out-of-bounds access.
        let string_length = u64::try_from(c_str.len()).unwrap();
        let string_length = string_length.checked_add(1).unwrap();
        if size < string_length {
            return Ok((false, string_length));
        }
        self.eval_context_mut()
            .write_bytes_ptr(ptr, c_str.iter().copied().chain(iter::once(0u8)))?;
        Ok((true, string_length))
    }

    /// Read a sequence of u16 until the first null terminator.
    fn read_wide_str(&self, mut ptr: Pointer<Option<Provenance>>) -> InterpResult<'tcx, Vec<u16>> {
        let this = self.eval_context_ref();
        let size2 = Size::from_bytes(2);
        let align2 = Align::from_bytes(2).unwrap();

        let mut wchars = Vec::new();
        loop {
            // FIXME: We are re-getting the allocation each time around the loop.
            // Would be nice if we could somehow "extend" an existing AllocRange.
            let alloc = this.get_ptr_alloc(ptr, size2, align2)?.unwrap(); // not a ZST, so we will get a result
            let wchar = alloc.read_integer(alloc_range(Size::ZERO, size2))?.to_u16()?;
            if wchar == 0 {
                break;
            } else {
                wchars.push(wchar);
                ptr = ptr.offset(size2, this)?;
            }
        }

        Ok(wchars)
    }

    /// Helper function to write a sequence of u16 with an added 0x0000-terminator, which is what
    /// the Windows APIs usually handle. This function returns `Ok((false, length))` without trying
    /// to write if `size` is not large enough to fit the contents of `os_string` plus a null
    /// terminator. It returns `Ok((true, length))` if the writing process was successful. The
    /// string length returned does include the null terminator. Length is measured in units of
    /// `u16.`
    fn write_wide_str(
        &mut self,
        wide_str: &[u16],
        ptr: Pointer<Option<Provenance>>,
        size: u64,
    ) -> InterpResult<'tcx, (bool, u64)> {
        // If `size` is smaller or equal than `bytes.len()`, writing `bytes` plus the required
        // 0x0000 terminator to memory would cause an out-of-bounds access.
        let string_length = u64::try_from(wide_str.len()).unwrap();
        let string_length = string_length.checked_add(1).unwrap();
        if size < string_length {
            return Ok((false, string_length));
        }

        // Store the UTF-16 string.
        let size2 = Size::from_bytes(2);
        let this = self.eval_context_mut();
        let mut alloc = this
            .get_ptr_alloc_mut(ptr, size2 * string_length, Align::from_bytes(2).unwrap())?
            .unwrap(); // not a ZST, so we will get a result
        for (offset, wchar) in wide_str.iter().copied().chain(iter::once(0x0000)).enumerate() {
            let offset = u64::try_from(offset).unwrap();
            alloc.write_scalar(alloc_range(size2 * offset, size2), Scalar::from_u16(wchar))?;
        }
        Ok((true, string_length))
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
        let Some(start_fn) = this.tcx.lang_items().start_fn() else {
            // no_std situations
            return false;
        };
        let frame = this.frame();
        // Make an attempt to get at the instance of the function this is inlined from.
        let instance: Option<_> = try {
            let scope = frame.current_source_info()?.scope;
            let inlined_parent = frame.body.source_scopes[scope].inlined_parent_scope?;
            let source = &frame.body.source_scopes[inlined_parent];
            source.inlined.expect("inlined_parent_scope points to scope without inline info").0
        };
        // Fall back to the instance of the function itself.
        let instance = instance.unwrap_or(frame.instance);
        // Now check if this is in the same crate as start_fn.
        // As a special exception we also allow unit tests from
        // <https://github.com/rust-lang/miri-test-libstd/tree/master/std_miri_test> to call these
        // shims.
        let frame_crate = this.tcx.def_path(instance.def_id()).krate;
        frame_crate == this.tcx.def_path(start_fn).krate
            || this.tcx.crate_name(frame_crate).as_str() == "std_miri_test"
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
            Ok(())
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
        if let Some((body, instance)) = self.eval_context_mut().lookup_exported_symbol(link_name)? {
            // If compiler-builtins is providing the symbol, then don't treat it as a clash.
            // We'll use our built-in implementation in `emulate_foreign_item_by_name` for increased
            // performance. Note that this means we won't catch any undefined behavior in
            // compiler-builtins when running other crates, but Miri can still be run on
            // compiler-builtins itself (or any crate that uses it as a normal dependency)
            if self.eval_context_ref().tcx.is_compiler_builtins(instance.def_id().krate) {
                return Ok(());
            }

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
        args: &'a [OpTy<'tcx, Provenance>],
    ) -> InterpResult<'tcx, &'a [OpTy<'tcx, Provenance>; N]>
    where
        &'a [OpTy<'tcx, Provenance>; N]: TryFrom<&'a [OpTy<'tcx, Provenance>]>,
    {
        self.check_abi_and_shim_symbol_clash(abi, exp_abi, link_name)?;
        check_arg_count(args)
    }

    /// Mark a machine allocation that was just created as immutable.
    fn mark_immutable(&mut self, mplace: &MemPlace<Provenance>) {
        let this = self.eval_context_mut();
        // This got just allocated, so there definitely is a pointer here.
        let provenance = mplace.ptr.into_pointer_or_addr().unwrap().provenance;
        this.alloc_mark_immutable(provenance.get_alloc_id().unwrap()).unwrap();
    }

    fn item_link_name(&self, def_id: DefId) -> Symbol {
        let tcx = self.eval_context_ref().tcx;
        match tcx.get_attrs(def_id, sym::link_name).filter_map(|a| a.value_str()).next() {
            Some(name) => name,
            None => tcx.item_name(def_id),
        }
    }
}

impl<'mir, 'tcx> MiriMachine<'mir, 'tcx> {
    /// Get the current span in the topmost function which is workspace-local and not
    /// `#[track_caller]`.
    /// This function is backed by a cache, and can be assumed to be very fast.
    /// It will work even when the stack is empty.
    pub fn current_span(&self) -> Span {
        self.top_user_relevant_frame()
            .map(|frame_idx| self.stack()[frame_idx].current_span())
            .unwrap_or(rustc_span::DUMMY_SP)
    }

    /// Returns the span of the *caller* of the current operation, again
    /// walking down the stack to find the closest frame in a local crate, if the caller of the
    /// current operation is not in a local crate.
    /// This is useful when we are processing something which occurs on function-entry and we want
    /// to point at the call to the function, not the function definition generally.
    pub fn caller_span(&self) -> Span {
        // We need to go down at least to the caller (len - 2), or however
        // far we have to go to find a frame in a local crate which is also not #[track_caller].
        let frame_idx = self.top_user_relevant_frame().unwrap();
        let frame_idx = cmp::min(frame_idx, self.stack().len().checked_sub(2).unwrap());
        self.stack()[frame_idx].current_span()
    }

    fn stack(&self) -> &[Frame<'mir, 'tcx, Provenance, machine::FrameExtra<'tcx>>] {
        self.threads.active_thread_stack()
    }

    fn top_user_relevant_frame(&self) -> Option<usize> {
        self.threads.active_thread_ref().top_user_relevant_frame()
    }

    /// This is the source of truth for the `is_user_relevant` flag in our `FrameExtra`.
    pub fn is_user_relevant(&self, frame: &Frame<'mir, 'tcx, Provenance>) -> bool {
        let def_id = frame.instance.def_id();
        (def_id.is_local() || self.local_crates.contains(&def_id.krate))
            && !frame.instance.def.requires_caller_location(self.tcx)
    }
}

/// Check that the number of args is what we expect.
pub fn check_arg_count<'a, 'tcx, const N: usize>(
    args: &'a [OpTy<'tcx, Provenance>],
) -> InterpResult<'tcx, &'a [OpTy<'tcx, Provenance>; N]>
where
    &'a [OpTy<'tcx, Provenance>; N]: TryFrom<&'a [OpTy<'tcx, Provenance>]>,
{
    if let Ok(ops) = args.try_into() {
        return Ok(ops);
    }
    throw_ub_format!("incorrect number of arguments: got {}, expected {}", args.len(), N)
}

pub fn isolation_abort_error<'tcx>(name: &str) -> InterpResult<'tcx> {
    throw_machine_stop!(TerminationInfo::UnsupportedInIsolation(format!(
        "{name} not available when isolation is enabled",
    )))
}

/// Retrieve the list of local crates that should have been passed by cargo-miri in
/// MIRI_LOCAL_CRATES and turn them into `CrateNum`s.
pub fn get_local_crates(tcx: TyCtxt<'_>) -> Vec<CrateNum> {
    // Convert the local crate names from the passed-in config into CrateNums so that they can
    // be looked up quickly during execution
    let local_crate_names = std::env::var("MIRI_LOCAL_CRATES")
        .map(|crates| crates.split(',').map(|krate| krate.to_string()).collect::<Vec<_>>())
        .unwrap_or_default();
    let mut local_crates = Vec::new();
    for &crate_num in tcx.crates(()) {
        let name = tcx.crate_name(crate_num);
        let name = name.as_str();
        if local_crate_names.iter().any(|local_name| local_name == name) {
            local_crates.push(crate_num);
        }
    }
    local_crates
}

/// Helper function used inside the shims of foreign functions to check that
/// `target_os` is a supported UNIX OS.
pub fn target_os_is_unix(target_os: &str) -> bool {
    matches!(target_os, "linux" | "macos" | "freebsd" | "android")
}
