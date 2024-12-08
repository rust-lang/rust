//! Implements calling functions from a native library.
use std::ops::Deref;

use libffi::high::call as ffi;
use libffi::low::CodePtr;
use rustc_abi::{BackendRepr, HasDataLayout, Size};
use rustc_middle::{
    mir::interpret::Pointer,
    ty::{self as ty, IntTy, UintTy},
};
use rustc_span::Symbol;

use crate::*;

impl<'tcx> EvalContextExtPriv<'tcx> for crate::MiriInterpCx<'tcx> {}
trait EvalContextExtPriv<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Call native host function and return the output as an immediate.
    fn call_native_with_args<'a>(
        &mut self,
        link_name: Symbol,
        dest: &MPlaceTy<'tcx>,
        ptr: CodePtr,
        libffi_args: Vec<libffi::high::Arg<'a>>,
    ) -> InterpResult<'tcx, ImmTy<'tcx>> {
        let this = self.eval_context_mut();

        // Call the function (`ptr`) with arguments `libffi_args`, and obtain the return value
        // as the specified primitive integer type
        let scalar = match dest.layout.ty.kind() {
            // ints
            ty::Int(IntTy::I8) => {
                // Unsafe because of the call to native code.
                // Because this is calling a C function it is not necessarily sound,
                // but there is no way around this and we've checked as much as we can.
                let x = unsafe { ffi::call::<i8>(ptr, libffi_args.as_slice()) };
                Scalar::from_i8(x)
            }
            ty::Int(IntTy::I16) => {
                let x = unsafe { ffi::call::<i16>(ptr, libffi_args.as_slice()) };
                Scalar::from_i16(x)
            }
            ty::Int(IntTy::I32) => {
                let x = unsafe { ffi::call::<i32>(ptr, libffi_args.as_slice()) };
                Scalar::from_i32(x)
            }
            ty::Int(IntTy::I64) => {
                let x = unsafe { ffi::call::<i64>(ptr, libffi_args.as_slice()) };
                Scalar::from_i64(x)
            }
            ty::Int(IntTy::Isize) => {
                let x = unsafe { ffi::call::<isize>(ptr, libffi_args.as_slice()) };
                Scalar::from_target_isize(x.try_into().unwrap(), this)
            }
            // uints
            ty::Uint(UintTy::U8) => {
                let x = unsafe { ffi::call::<u8>(ptr, libffi_args.as_slice()) };
                Scalar::from_u8(x)
            }
            ty::Uint(UintTy::U16) => {
                let x = unsafe { ffi::call::<u16>(ptr, libffi_args.as_slice()) };
                Scalar::from_u16(x)
            }
            ty::Uint(UintTy::U32) => {
                let x = unsafe { ffi::call::<u32>(ptr, libffi_args.as_slice()) };
                Scalar::from_u32(x)
            }
            ty::Uint(UintTy::U64) => {
                let x = unsafe { ffi::call::<u64>(ptr, libffi_args.as_slice()) };
                Scalar::from_u64(x)
            }
            ty::Uint(UintTy::Usize) => {
                let x = unsafe { ffi::call::<usize>(ptr, libffi_args.as_slice()) };
                Scalar::from_target_usize(x.try_into().unwrap(), this)
            }
            // Functions with no declared return type (i.e., the default return)
            // have the output_type `Tuple([])`.
            ty::Tuple(t_list) if t_list.len() == 0 => {
                unsafe { ffi::call::<()>(ptr, libffi_args.as_slice()) };
                return interp_ok(ImmTy::uninit(dest.layout));
            }
            ty::RawPtr(..) => {
                let x = unsafe { ffi::call::<*const ()>(ptr, libffi_args.as_slice()) };
                let ptr = Pointer::new(Provenance::Wildcard, Size::from_bytes(x.addr()));
                Scalar::from_pointer(ptr, this)
            }
            _ => throw_unsup_format!("unsupported return type for native call: {:?}", link_name),
        };
        interp_ok(ImmTy::from_scalar(scalar, dest.layout))
    }

    /// Get the pointer to the function of the specified name in the shared object file,
    /// if it exists. The function must be in the shared object file specified: we do *not*
    /// return pointers to functions in dependencies of the library.  
    fn get_func_ptr_explicitly_from_lib(&mut self, link_name: Symbol) -> Option<CodePtr> {
        let this = self.eval_context_mut();
        // Try getting the function from the shared library.
        // On windows `_lib_path` will be unused, hence the name starting with `_`.
        let (lib, _lib_path) = this.machine.native_lib.as_ref().unwrap();
        let func: libloading::Symbol<'_, unsafe extern "C" fn()> = unsafe {
            match lib.get(link_name.as_str().as_bytes()) {
                Ok(x) => x,
                Err(_) => {
                    return None;
                }
            }
        };

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
        let mut info = std::mem::MaybeUninit::<libc::Dl_info>::uninit();
        unsafe {
            if libc::dladdr(*func.deref() as *const _, info.as_mut_ptr()) != 0 {
                if std::ffi::CStr::from_ptr(info.assume_init().dli_fname).to_str().unwrap()
                    != _lib_path.to_str().unwrap()
                {
                    return None;
                }
            }
        }
        // Return a pointer to the function.
        Some(CodePtr(*func.deref() as *mut _))
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

        // Get the function arguments, and convert them to `libffi`-compatible form.
        let mut libffi_args = Vec::<CArg>::with_capacity(args.len());
        for arg in args.iter() {
            if !matches!(arg.layout.backend_repr, BackendRepr::Scalar(_)) {
                throw_unsup_format!("only scalar argument types are support for native calls")
            }
            let imm = this.read_immediate(arg)?;
            libffi_args.push(imm_to_carg(&imm, this)?);
            // If we are passing a pointer, prepare the memory it points to.
            if matches!(arg.layout.ty.kind(), ty::RawPtr(..)) {
                let ptr = imm.to_scalar().to_pointer(this)?;
                let Some(prov) = ptr.provenance else {
                    // Pointer without provenance may not access any memory.
                    continue;
                };
                // We use `get_alloc_id` for its best-effort behaviour with Wildcard provenance.
                let Some(alloc_id) = prov.get_alloc_id() else {
                    // Wildcard pointer, whatever it points to must be already exposed.
                    continue;
                };
                this.prepare_for_native_call(alloc_id, prov)?;
            }
        }
        
        // FIXME: In the future, we should also call `prepare_for_native_call` on all previously
        // exposed allocations, since C may access any of them.

        // Convert them to `libffi::high::Arg` type.
        let libffi_args = libffi_args
            .iter()
            .map(|arg| arg.arg_downcast())
            .collect::<Vec<libffi::high::Arg<'_>>>();

        // Call the function and store output, depending on return type in the function signature.
        let ret = this.call_native_with_args(link_name, dest, code_ptr, libffi_args)?;
        this.write_immediate(*ret, dest)?;
        interp_ok(true)
    }
}

#[derive(Debug, Clone)]
/// Enum of supported arguments to external C functions.
// We introduce this enum instead of just calling `ffi::arg` and storing a list
// of `libffi::high::Arg` directly, because the `libffi::high::Arg` just wraps a reference
// to the value it represents: https://docs.rs/libffi/latest/libffi/high/call/struct.Arg.html
// and we need to store a copy of the value, and pass a reference to this copy to C instead.
enum CArg {
    /// 8-bit signed integer.
    Int8(i8),
    /// 16-bit signed integer.
    Int16(i16),
    /// 32-bit signed integer.
    Int32(i32),
    /// 64-bit signed integer.
    Int64(i64),
    /// isize.
    ISize(isize),
    /// 8-bit unsigned integer.
    UInt8(u8),
    /// 16-bit unsigned integer.
    UInt16(u16),
    /// 32-bit unsigned integer.
    UInt32(u32),
    /// 64-bit unsigned integer.
    UInt64(u64),
    /// usize.
    USize(usize),
    /// Raw pointer, stored as C's `void*`.
    RawPtr(*mut std::ffi::c_void),
}

impl<'a> CArg {
    /// Convert a `CArg` to a `libffi` argument type.
    fn arg_downcast(&'a self) -> libffi::high::Arg<'a> {
        match self {
            CArg::Int8(i) => ffi::arg(i),
            CArg::Int16(i) => ffi::arg(i),
            CArg::Int32(i) => ffi::arg(i),
            CArg::Int64(i) => ffi::arg(i),
            CArg::ISize(i) => ffi::arg(i),
            CArg::UInt8(i) => ffi::arg(i),
            CArg::UInt16(i) => ffi::arg(i),
            CArg::UInt32(i) => ffi::arg(i),
            CArg::UInt64(i) => ffi::arg(i),
            CArg::USize(i) => ffi::arg(i),
            CArg::RawPtr(i) => ffi::arg(i),
        }
    }
}

/// Extract the scalar value from the result of reading a scalar from the machine,
/// and convert it to a `CArg`.
fn imm_to_carg<'tcx>(v: &ImmTy<'tcx>, cx: &impl HasDataLayout) -> InterpResult<'tcx, CArg> {
    interp_ok(match v.layout.ty.kind() {
        // If the primitive provided can be converted to a type matching the type pattern
        // then create a `CArg` of this primitive value with the corresponding `CArg` constructor.
        // the ints
        ty::Int(IntTy::I8) => CArg::Int8(v.to_scalar().to_i8()?),
        ty::Int(IntTy::I16) => CArg::Int16(v.to_scalar().to_i16()?),
        ty::Int(IntTy::I32) => CArg::Int32(v.to_scalar().to_i32()?),
        ty::Int(IntTy::I64) => CArg::Int64(v.to_scalar().to_i64()?),
        ty::Int(IntTy::Isize) =>
            CArg::ISize(v.to_scalar().to_target_isize(cx)?.try_into().unwrap()),
        // the uints
        ty::Uint(UintTy::U8) => CArg::UInt8(v.to_scalar().to_u8()?),
        ty::Uint(UintTy::U16) => CArg::UInt16(v.to_scalar().to_u16()?),
        ty::Uint(UintTy::U32) => CArg::UInt32(v.to_scalar().to_u32()?),
        ty::Uint(UintTy::U64) => CArg::UInt64(v.to_scalar().to_u64()?),
        ty::Uint(UintTy::Usize) =>
            CArg::USize(v.to_scalar().to_target_usize(cx)?.try_into().unwrap()),
        ty::RawPtr(..) => {
            let s = v.to_scalar().to_pointer(cx)?.addr();
            // This relies on the `expose_provenance` in `addr_from_alloc_id`.
            CArg::RawPtr(std::ptr::with_exposed_provenance_mut(s.bytes_usize()))
        }
        _ => throw_unsup_format!("unsupported argument type for native call: {}", v.layout.ty),
    })
}
