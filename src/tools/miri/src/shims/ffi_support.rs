use libffi::{high::call as ffi, low::CodePtr};
use std::ops::Deref;

use rustc_middle::ty::{self as ty, IntTy, Ty, UintTy};
use rustc_span::Symbol;
use rustc_target::abi::HasDataLayout;

use crate::*;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}

pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    /// Extract the scalar value from the result of reading a scalar from the machine,
    /// and convert it to a `CArg`.
    fn scalar_to_carg(
        k: Scalar<Provenance>,
        arg_type: Ty<'tcx>,
        cx: &impl HasDataLayout,
    ) -> InterpResult<'tcx, CArg> {
        match arg_type.kind() {
            // If the primitive provided can be converted to a type matching the type pattern
            // then create a `CArg` of this primitive value with the corresponding `CArg` constructor.
            // the ints
            ty::Int(IntTy::I8) => {
                return Ok(CArg::Int8(k.to_i8()?));
            }
            ty::Int(IntTy::I16) => {
                return Ok(CArg::Int16(k.to_i16()?));
            }
            ty::Int(IntTy::I32) => {
                return Ok(CArg::Int32(k.to_i32()?));
            }
            ty::Int(IntTy::I64) => {
                return Ok(CArg::Int64(k.to_i64()?));
            }
            ty::Int(IntTy::Isize) => {
                // This will fail if host != target, but then the entire FFI thing probably won't work well
                // in that situation.
                return Ok(CArg::ISize(k.to_target_isize(cx)?.try_into().unwrap()));
            }
            // the uints
            ty::Uint(UintTy::U8) => {
                return Ok(CArg::UInt8(k.to_u8()?));
            }
            ty::Uint(UintTy::U16) => {
                return Ok(CArg::UInt16(k.to_u16()?));
            }
            ty::Uint(UintTy::U32) => {
                return Ok(CArg::UInt32(k.to_u32()?));
            }
            ty::Uint(UintTy::U64) => {
                return Ok(CArg::UInt64(k.to_u64()?));
            }
            ty::Uint(UintTy::Usize) => {
                // This will fail if host != target, but then the entire FFI thing probably won't work well
                // in that situation.
                return Ok(CArg::USize(k.to_target_usize(cx)?.try_into().unwrap()));
            }
            _ => {}
        }
        // If no primitives were returned then we have an unsupported type.
        throw_unsup_format!(
            "unsupported scalar argument type to external C function: {:?}",
            arg_type
        );
    }

    /// Call external C function and
    /// store output, depending on return type in the function signature.
    fn call_external_c_and_store_return<'a>(
        &mut self,
        link_name: Symbol,
        dest: &PlaceTy<'tcx, Provenance>,
        ptr: CodePtr,
        libffi_args: Vec<libffi::high::Arg<'a>>,
    ) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();

        // Unsafe because of the call to external C code.
        // Because this is calling a C function it is not necessarily sound,
        // but there is no way around this and we've checked as much as we can.
        unsafe {
            // If the return type of a function is a primitive integer type,
            // then call the function (`ptr`) with arguments `libffi_args`, store the return value as the specified
            // primitive integer type, and then write this value out to the miri memory as an integer.
            match dest.layout.ty.kind() {
                // ints
                ty::Int(IntTy::I8) => {
                    let x = ffi::call::<i8>(ptr, libffi_args.as_slice());
                    this.write_int(x, dest)?;
                    return Ok(());
                }
                ty::Int(IntTy::I16) => {
                    let x = ffi::call::<i16>(ptr, libffi_args.as_slice());
                    this.write_int(x, dest)?;
                    return Ok(());
                }
                ty::Int(IntTy::I32) => {
                    let x = ffi::call::<i32>(ptr, libffi_args.as_slice());
                    this.write_int(x, dest)?;
                    return Ok(());
                }
                ty::Int(IntTy::I64) => {
                    let x = ffi::call::<i64>(ptr, libffi_args.as_slice());
                    this.write_int(x, dest)?;
                    return Ok(());
                }
                ty::Int(IntTy::Isize) => {
                    let x = ffi::call::<isize>(ptr, libffi_args.as_slice());
                    // `isize` doesn't `impl Into<i128>`, so convert manually.
                    // Convert to `i64` since this covers both 32- and 64-bit machines.
                    this.write_int(i64::try_from(x).unwrap(), dest)?;
                    return Ok(());
                }
                // uints
                ty::Uint(UintTy::U8) => {
                    let x = ffi::call::<u8>(ptr, libffi_args.as_slice());
                    this.write_int(x, dest)?;
                    return Ok(());
                }
                ty::Uint(UintTy::U16) => {
                    let x = ffi::call::<u16>(ptr, libffi_args.as_slice());
                    this.write_int(x, dest)?;
                    return Ok(());
                }
                ty::Uint(UintTy::U32) => {
                    let x = ffi::call::<u32>(ptr, libffi_args.as_slice());
                    this.write_int(x, dest)?;
                    return Ok(());
                }
                ty::Uint(UintTy::U64) => {
                    let x = ffi::call::<u64>(ptr, libffi_args.as_slice());
                    this.write_int(x, dest)?;
                    return Ok(());
                }
                ty::Uint(UintTy::Usize) => {
                    let x = ffi::call::<usize>(ptr, libffi_args.as_slice());
                    // `usize` doesn't `impl Into<i128>`, so convert manually.
                    // Convert to `u64` since this covers both 32- and 64-bit machines.
                    this.write_int(u64::try_from(x).unwrap(), dest)?;
                    return Ok(());
                }
                // Functions with no declared return type (i.e., the default return)
                // have the output_type `Tuple([])`.
                ty::Tuple(t_list) =>
                    if t_list.len() == 0 {
                        ffi::call::<()>(ptr, libffi_args.as_slice());
                        return Ok(());
                    },
                _ => {}
            }
            // FIXME ellen! deal with all the other return types
            throw_unsup_format!("unsupported return type to external C function: {:?}", link_name);
        }
    }

    /// Get the pointer to the function of the specified name in the shared object file,
    /// if it exists. The function must be in the shared object file specified: we do *not*
    /// return pointers to functions in dependencies of the library.  
    fn get_func_ptr_explicitly_from_lib(&mut self, link_name: Symbol) -> Option<CodePtr> {
        let this = self.eval_context_mut();
        // Try getting the function from the shared library.
        // On windows `_lib_path` will be unused, hence the name starting with `_`.
        let (lib, _lib_path) = this.machine.external_so_lib.as_ref().unwrap();
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
        // No `libc::dladdr` on windows.
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

    /// Call specified external C function, with supplied arguments.
    /// Need to convert all the arguments from their hir representations to
    /// a form compatible with C (through `libffi` call).
    /// Then, convert return from the C call into a corresponding form that
    /// can be stored in Miri internal memory.
    fn call_external_c_fct(
        &mut self,
        link_name: Symbol,
        dest: &PlaceTy<'tcx, Provenance>,
        args: &[OpTy<'tcx, Provenance>],
    ) -> InterpResult<'tcx, bool> {
        // Get the pointer to the function in the shared object file if it exists.
        let code_ptr = match self.get_func_ptr_explicitly_from_lib(link_name) {
            Some(ptr) => ptr,
            None => {
                // Shared object file does not export this function -- try the shims next.
                return Ok(false);
            }
        };

        let this = self.eval_context_mut();

        // Get the function arguments, and convert them to `libffi`-compatible form.
        let mut libffi_args = Vec::<CArg>::with_capacity(args.len());
        for cur_arg in args.iter() {
            libffi_args.push(Self::scalar_to_carg(
                this.read_scalar(cur_arg)?,
                cur_arg.layout.ty,
                this,
            )?);
        }

        // Convert them to `libffi::high::Arg` type.
        let libffi_args = libffi_args
            .iter()
            .map(|cur_arg| cur_arg.arg_downcast())
            .collect::<Vec<libffi::high::Arg<'_>>>();

        // Call the function and store output, depending on return type in the function signature.
        self.call_external_c_and_store_return(link_name, dest, code_ptr, libffi_args)?;
        Ok(true)
    }
}

#[derive(Debug, Clone)]
/// Enum of supported arguments to external C functions.
// We introduce this enum instead of just calling `ffi::arg` and storing a list
// of `libffi::high::Arg` directly, because the `libffi::high::Arg` just wraps a reference
// to the value it represents: https://docs.rs/libffi/latest/libffi/high/call/struct.Arg.html
// and we need to store a copy of the value, and pass a reference to this copy to C instead.
pub enum CArg {
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
        }
    }
}
