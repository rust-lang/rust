//! Implements calling functions from a native library.

use std::ops::Deref;

use libffi::low::CodePtr;
use libffi::middle::Type as FfiType;
use rustc_abi::{HasDataLayout, Size};
use rustc_middle::ty::{self as ty, IntTy, Ty, UintTy};
use rustc_span::Symbol;
use serde::{Deserialize, Serialize};

mod ffi;

#[cfg_attr(
    not(all(
        target_os = "linux",
        target_env = "gnu",
        any(target_arch = "x86", target_arch = "x86_64")
    )),
    path = "trace/stub.rs"
)]
pub mod trace;

use self::ffi::OwnedArg;
use crate::*;

/// The final results of an FFI trace, containing every relevant event detected
/// by the tracer.
#[derive(Serialize, Deserialize, Debug)]
pub struct MemEvents {
    /// An list of memory accesses that occurred, in the order they occurred in.
    pub acc_events: Vec<AccessEvent>,
}

/// A single memory access.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum AccessEvent {
    /// A read occurred on this memory range.
    Read(AccessRange),
    /// A write may have occurred on this memory range.
    /// Some instructions *may* write memory without *always* doing that,
    /// so this can be an over-approximation.
    /// The range info, however, is reliable if the access did happen.
    /// If the second field is true, the access definitely happened.
    Write(AccessRange, bool),
}

impl AccessEvent {
    fn get_range(&self) -> AccessRange {
        match self {
            AccessEvent::Read(access_range) => access_range.clone(),
            AccessEvent::Write(access_range, _) => access_range.clone(),
        }
    }
}

/// The memory touched by a given access.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AccessRange {
    /// The base address in memory where an access occurred.
    pub addr: usize,
    /// The number of bytes affected from the base.
    pub size: usize,
}

impl AccessRange {
    fn end(&self) -> usize {
        self.addr.strict_add(self.size)
    }
}

impl<'tcx> EvalContextExtPriv<'tcx> for crate::MiriInterpCx<'tcx> {}
trait EvalContextExtPriv<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Call native host function and return the output as an immediate.
    fn call_native_with_args(
        &mut self,
        link_name: Symbol,
        dest: &MPlaceTy<'tcx>,
        fun: CodePtr,
        libffi_args: &mut [OwnedArg],
    ) -> InterpResult<'tcx, (crate::ImmTy<'tcx>, Option<MemEvents>)> {
        let this = self.eval_context_mut();
        #[cfg(target_os = "linux")]
        let alloc = this.machine.allocator.as_ref().unwrap();
        #[cfg(not(target_os = "linux"))]
        // Placeholder value.
        let alloc = ();

        trace::Supervisor::do_ffi(alloc, || {
            // Call the function (`ptr`) with arguments `libffi_args`, and obtain the return value
            // as the specified primitive integer type
            let scalar = match dest.layout.ty.kind() {
                // ints
                ty::Int(IntTy::I8) => {
                    // Unsafe because of the call to native code.
                    // Because this is calling a C function it is not necessarily sound,
                    // but there is no way around this and we've checked as much as we can.
                    let x = unsafe { ffi::call::<i8>(fun, libffi_args) };
                    Scalar::from_i8(x)
                }
                ty::Int(IntTy::I16) => {
                    let x = unsafe { ffi::call::<i16>(fun, libffi_args) };
                    Scalar::from_i16(x)
                }
                ty::Int(IntTy::I32) => {
                    let x = unsafe { ffi::call::<i32>(fun, libffi_args) };
                    Scalar::from_i32(x)
                }
                ty::Int(IntTy::I64) => {
                    let x = unsafe { ffi::call::<i64>(fun, libffi_args) };
                    Scalar::from_i64(x)
                }
                ty::Int(IntTy::Isize) => {
                    let x = unsafe { ffi::call::<isize>(fun, libffi_args) };
                    Scalar::from_target_isize(x.try_into().unwrap(), this)
                }
                // uints
                ty::Uint(UintTy::U8) => {
                    let x = unsafe { ffi::call::<u8>(fun, libffi_args) };
                    Scalar::from_u8(x)
                }
                ty::Uint(UintTy::U16) => {
                    let x = unsafe { ffi::call::<u16>(fun, libffi_args) };
                    Scalar::from_u16(x)
                }
                ty::Uint(UintTy::U32) => {
                    let x = unsafe { ffi::call::<u32>(fun, libffi_args) };
                    Scalar::from_u32(x)
                }
                ty::Uint(UintTy::U64) => {
                    let x = unsafe { ffi::call::<u64>(fun, libffi_args) };
                    Scalar::from_u64(x)
                }
                ty::Uint(UintTy::Usize) => {
                    let x = unsafe { ffi::call::<usize>(fun, libffi_args) };
                    Scalar::from_target_usize(x.try_into().unwrap(), this)
                }
                // Functions with no declared return type (i.e., the default return)
                // have the output_type `Tuple([])`.
                ty::Tuple(t_list) if (*t_list).deref().is_empty() => {
                    unsafe { ffi::call::<()>(fun, libffi_args) };
                    return interp_ok(ImmTy::uninit(dest.layout));
                }
                ty::RawPtr(..) => {
                    let x = unsafe { ffi::call::<*const ()>(fun, libffi_args) };
                    let ptr = StrictPointer::new(Provenance::Wildcard, Size::from_bytes(x.addr()));
                    Scalar::from_pointer(ptr, this)
                }
                _ =>
                    return Err(err_unsup_format!(
                        "unsupported return type for native call: {:?}",
                        link_name
                    ))
                    .into(),
            };
            interp_ok(ImmTy::from_scalar(scalar, dest.layout))
        })
    }

    /// Get the pointer to the function of the specified name in the shared object file,
    /// if it exists. The function must be in one of the shared object files specified:
    /// we do *not* return pointers to functions in dependencies of libraries.
    fn get_func_ptr_explicitly_from_lib(&mut self, link_name: Symbol) -> Option<CodePtr> {
        let this = self.eval_context_mut();
        // Try getting the function from one of the shared libraries.
        for (lib, lib_path) in &this.machine.native_lib {
            let Ok(func): Result<libloading::Symbol<'_, unsafe extern "C" fn()>, _> =
                (unsafe { lib.get(link_name.as_str().as_bytes()) })
            else {
                continue;
            };
            #[expect(clippy::as_conversions)] // fn-ptr to raw-ptr cast needs `as`.
            let fn_ptr = *func.deref() as *mut std::ffi::c_void;

            // FIXME: this is a hack!
            // The `libloading` crate will automatically load system libraries like `libc`.
            // On linux `libloading` is based on `dlsym`: https://docs.rs/libloading/0.7.3/src/libloading/os/unix/mod.rs.html#202
            // and `dlsym`(https://linux.die.net/man/3/dlsym) looks through the dependency tree of the
            // library if it can't find the symbol in the library itself.
            // So, in order to check if the function was actually found in the specified
            // `machine.external_so_lib` we need to check its `dli_fname` and compare it to
            // the specified SO file path.
            // This code is a reimplementation of the mechanism for getting `dli_fname` in `libloading`,
            // from: https://docs.rs/libloading/0.7.3/src/libloading/os/unix/mod.rs.html#411
            // using the `libc` crate where this interface is public.
            let mut info = std::mem::MaybeUninit::<libc::Dl_info>::zeroed();
            unsafe {
                let res = libc::dladdr(fn_ptr, info.as_mut_ptr());
                assert!(res != 0, "failed to load info about function we already loaded");
                let info = info.assume_init();
                #[cfg(target_os = "cygwin")]
                let fname_ptr = info.dli_fname.as_ptr();
                #[cfg(not(target_os = "cygwin"))]
                let fname_ptr = info.dli_fname;
                assert!(!fname_ptr.is_null());
                if std::ffi::CStr::from_ptr(fname_ptr).to_str().unwrap()
                    != lib_path.to_str().unwrap()
                {
                    // The function is not actually in this .so, check the next one.
                    continue;
                }
            }

            // Return a pointer to the function.
            return Some(CodePtr(fn_ptr));
        }
        None
    }

    /// Applies the `events` to Miri's internal state. The event vector must be
    /// ordered sequentially by when the accesses happened, and the sizes are
    /// assumed to be exact.
    fn tracing_apply_accesses(&mut self, events: MemEvents) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        for evt in events.acc_events {
            let evt_rg = evt.get_range();
            // LLVM at least permits vectorising accesses to adjacent allocations,
            // so we cannot assume 1 access = 1 allocation. :(
            let mut rg = evt_rg.addr..evt_rg.end();
            while let Some(curr) = rg.next() {
                let Some(alloc_id) = this.alloc_id_from_addr(
                    curr.to_u64(),
                    rg.len().try_into().unwrap(),
                    /* only_exposed_allocations */ true,
                ) else {
                    throw_ub_format!("Foreign code did an out-of-bounds access!")
                };
                let alloc = this.get_alloc_raw(alloc_id)?;
                // The logical and physical address of the allocation coincide, so we can use
                // this instead of `addr_from_alloc_id`.
                let alloc_addr = alloc.get_bytes_unchecked_raw().addr();

                // Determine the range inside the allocation that this access covers. This range is
                // in terms of offsets from the start of `alloc`. The start of the overlap range
                // will be `curr`; the end will be the minimum of the end of the allocation and the
                // end of the access' range.
                let overlap = curr.strict_sub(alloc_addr)
                    ..std::cmp::min(alloc.len(), rg.end.strict_sub(alloc_addr));
                // Skip forward however many bytes of the access are contained in the current
                // allocation, subtracting 1 since the overlap range includes the current addr
                // that was already popped off of the range.
                rg.advance_by(overlap.len().strict_sub(1)).unwrap();

                match evt {
                    AccessEvent::Read(_) => {
                        // If a provenance was read by the foreign code, expose it.
                        for prov in alloc.provenance().get_range(this, overlap.into()) {
                            this.expose_provenance(prov)?;
                        }
                    }
                    AccessEvent::Write(_, certain) => {
                        // Sometimes we aren't certain if a write happened, in which case we
                        // only initialise that data if the allocation is mutable.
                        if certain || alloc.mutability.is_mut() {
                            let (alloc, cx) = this.get_alloc_raw_mut(alloc_id)?;
                            alloc.process_native_write(
                                &cx.tcx,
                                Some(AllocRange {
                                    start: Size::from_bytes(overlap.start),
                                    size: Size::from_bytes(overlap.len()),
                                }),
                            )
                        }
                    }
                }
            }
        }

        interp_ok(())
    }

    /// Extract the value from the result of reading an operand from the machine
    /// and convert it to a `OwnedArg`.
    fn op_to_ffi_arg(&self, v: &OpTy<'tcx>, tracing: bool) -> InterpResult<'tcx, OwnedArg> {
        let this = self.eval_context_ref();

        // This should go first so that we emit unsupported before doing a bunch
        // of extra work for types that aren't supported yet.
        let ty = this.ty_to_ffitype(v.layout.ty)?;

        // Helper to print a warning when a pointer is shared with the native code.
        let expose = |prov: Provenance| -> InterpResult<'tcx> {
            // The first time this happens, print a warning.
            if !this.machine.native_call_mem_warned.replace(true) {
                // Newly set, so first time we get here.
                this.emit_diagnostic(NonHaltingDiagnostic::NativeCallSharedMem { tracing });
            }

            this.expose_provenance(prov)?;
            interp_ok(())
        };

        // Compute the byte-level representation of the argument. If there's a pointer in there, we
        // expose it inside the AM. Later in `visit_reachable_allocs`, the "meta"-level provenance
        // for accessing the pointee gets exposed; this is crucial to justify the C code effectively
        // casting the integer in `byte` to a pointer and using that.
        let bytes = match v.as_mplace_or_imm() {
            either::Either::Left(mplace) => {
                // Get the alloc id corresponding to this mplace, alongside
                // a pointer that's offset to point to this particular
                // mplace (not one at the base addr of the allocation).
                let sz = mplace.layout.size.bytes_usize();
                if sz == 0 {
                    throw_unsup_format!("attempting to pass a ZST over FFI");
                }
                let (id, ofs, _) = this.ptr_get_alloc_id(mplace.ptr(), sz.try_into().unwrap())?;
                let ofs = ofs.bytes_usize();
                let range = ofs..ofs.strict_add(sz);
                // Expose all provenances in the allocation within the byte range of the struct, if
                // any. These pointers are being directly passed to native code by-value.
                let alloc = this.get_alloc_raw(id)?;
                for prov in alloc.provenance().get_range(this, range.clone().into()) {
                    expose(prov)?;
                }
                // Read the bytes that make up this argument. We cannot use the normal getter as
                // those would fail if any part of the argument is uninitialized. Native code
                // is kind of outside the interpreter, after all...
                Box::from(alloc.inspect_with_uninit_and_ptr_outside_interpreter(range))
            }
            either::Either::Right(imm) => {
                let mut bytes: Box<[u8]> = vec![0; imm.layout.size.bytes_usize()].into();

                // A little helper to write scalars to our byte array.
                let mut write_scalar = |this: &MiriInterpCx<'tcx>, sc: Scalar, pos: usize| {
                    // If a scalar is a pointer, then expose its provenance.
                    if let interpret::Scalar::Ptr(p, _) = sc {
                        expose(p.provenance)?;
                    }
                    write_target_uint(
                        this.data_layout().endian,
                        &mut bytes[pos..][..sc.size().bytes_usize()],
                        sc.to_scalar_int()?.to_bits_unchecked(),
                    )
                    .unwrap();
                    interp_ok(())
                };

                // Write the scalar into the `bytes` buffer.
                match *imm {
                    Immediate::Scalar(sc) => write_scalar(this, sc, 0)?,
                    Immediate::ScalarPair(sc_first, sc_second) => {
                        // The first scalar has an offset of zero; compute the offset of the 2nd.
                        let ofs_second = {
                            let rustc_abi::BackendRepr::ScalarPair(a, b) = imm.layout.backend_repr
                            else {
                                span_bug!(
                                    this.cur_span(),
                                    "op_to_ffi_arg: invalid scalar pair layout: {:#?}",
                                    imm.layout
                                )
                            };
                            a.size(this).align_to(b.align(this).abi).bytes_usize()
                        };

                        write_scalar(this, sc_first, 0)?;
                        write_scalar(this, sc_second, ofs_second)?;
                    }
                    Immediate::Uninit => {
                        // Nothing to write.
                    }
                }

                bytes
            }
        };
        interp_ok(OwnedArg::new(ty, bytes))
    }

    /// Parses an ADT to construct the matching libffi type.
    fn adt_to_ffitype(
        &self,
        orig_ty: Ty<'_>,
        adt_def: ty::AdtDef<'tcx>,
        args: &'tcx ty::List<ty::GenericArg<'tcx>>,
    ) -> InterpResult<'tcx, FfiType> {
        // TODO: Certain non-C reprs should be okay also.
        if !adt_def.repr().c() {
            throw_unsup_format!("passing a non-#[repr(C)] struct over FFI: {orig_ty}")
        }
        // TODO: unions, etc.
        if !adt_def.is_struct() {
            throw_unsup_format!(
                "unsupported argument type for native call: {orig_ty} is an enum or union"
            );
        }

        let this = self.eval_context_ref();
        let mut fields = vec![];
        for field in &adt_def.non_enum_variant().fields {
            fields.push(this.ty_to_ffitype(field.ty(*this.tcx, args))?);
        }

        interp_ok(FfiType::structure(fields))
    }

    /// Gets the matching libffi type for a given Ty.
    fn ty_to_ffitype(&self, ty: Ty<'tcx>) -> InterpResult<'tcx, FfiType> {
        interp_ok(match ty.kind() {
            ty::Int(IntTy::I8) => FfiType::i8(),
            ty::Int(IntTy::I16) => FfiType::i16(),
            ty::Int(IntTy::I32) => FfiType::i32(),
            ty::Int(IntTy::I64) => FfiType::i64(),
            ty::Int(IntTy::Isize) => FfiType::isize(),
            // the uints
            ty::Uint(UintTy::U8) => FfiType::u8(),
            ty::Uint(UintTy::U16) => FfiType::u16(),
            ty::Uint(UintTy::U32) => FfiType::u32(),
            ty::Uint(UintTy::U64) => FfiType::u64(),
            ty::Uint(UintTy::Usize) => FfiType::usize(),
            ty::RawPtr(..) => FfiType::pointer(),
            ty::Adt(adt_def, args) => self.adt_to_ffitype(ty, *adt_def, args)?,
            _ => throw_unsup_format!("unsupported argument type for native call: {}", ty),
        })
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Call the native host function, with supplied arguments.
    /// Needs to convert all the arguments from their Miri representations to
    /// a native form (through `libffi` call).
    /// Then, convert the return value from the native form into something that
    /// can be stored in Miri's internal memory.
    fn call_native_fn(
        &mut self,
        link_name: Symbol,
        dest: &MPlaceTy<'tcx>,
        args: &[OpTy<'tcx>],
    ) -> InterpResult<'tcx, bool> {
        let this = self.eval_context_mut();
        // Get the pointer to the function in the shared object file if it exists.
        let code_ptr = match this.get_func_ptr_explicitly_from_lib(link_name) {
            Some(ptr) => ptr,
            None => {
                // Shared object file does not export this function -- try the shims next.
                return interp_ok(false);
            }
        };

        // Do we have ptrace?
        let tracing = trace::Supervisor::is_enabled();

        // Get the function arguments, copy them, and prepare the type descriptions.
        let mut libffi_args = Vec::<OwnedArg>::with_capacity(args.len());
        for arg in args.iter() {
            libffi_args.push(this.op_to_ffi_arg(arg, tracing)?);
        }

        // Prepare all exposed memory (both previously exposed, and just newly exposed since a
        // pointer was passed as argument). Uninitialised memory is left as-is, but any data
        // exposed this way is garbage anyway.
        this.visit_reachable_allocs(this.exposed_allocs(), |this, alloc_id, info| {
            // If there is no data behind this pointer, skip this.
            if !matches!(info.kind, AllocKind::LiveData) {
                return interp_ok(());
            }
            // It's okay to get raw access, what we do does not correspond to any actual
            // AM operation, it just approximates the state to account for the native call.
            let alloc = this.get_alloc_raw(alloc_id)?;
            // Also expose the provenance of the interpreter-level allocation, so it can
            // be read by FFI. The `black_box` is defensive programming as LLVM likes
            // to (incorrectly) optimize away ptr2int casts whose result is unused.
            std::hint::black_box(alloc.get_bytes_unchecked_raw().expose_provenance());

            if !tracing {
                // Expose all provenances in this allocation, since the native code can do
                // $whatever. Can be skipped when tracing; in that case we'll expose just the
                // actually-read parts later.
                for prov in alloc.provenance().provenances() {
                    this.expose_provenance(prov)?;
                }
            }

            // Prepare for possible write from native code if mutable.
            if info.mutbl.is_mut() {
                let (alloc, cx) = this.get_alloc_raw_mut(alloc_id)?;
                // These writes could initialize everything and wreck havoc with the pointers.
                // We can skip that when tracing; in that case we'll later do that only for the
                // memory that got actually written.
                if !tracing {
                    alloc.process_native_write(&cx.tcx, None);
                }
                // Also expose *mutable* provenance for the interpreter-level allocation.
                std::hint::black_box(alloc.get_bytes_unchecked_raw_mut().expose_provenance());
            }

            interp_ok(())
        })?;

        // Call the function and store output, depending on return type in the function signature.
        let (ret, maybe_memevents) =
            this.call_native_with_args(link_name, dest, code_ptr, &mut libffi_args)?;

        if tracing {
            this.tracing_apply_accesses(maybe_memevents.unwrap())?;
        }

        this.write_immediate(*ret, dest)?;
        interp_ok(true)
    }
}
