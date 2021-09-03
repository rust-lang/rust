use std::iter;

use rustc_middle::mir;
use rustc_span::Symbol;
use rustc_target::abi::Size;
use rustc_target::spec::abi::Abi;

use crate::*;
use shims::foreign_items::EmulateByNameResult;
use shims::windows::sync::EvalContextExt as _;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn emulate_foreign_item_by_name(
        &mut self,
        link_name: Symbol,
        abi: Abi,
        args: &[OpTy<'tcx, Tag>],
        dest: &PlaceTy<'tcx, Tag>,
        _ret: mir::BasicBlock,
    ) -> InterpResult<'tcx, EmulateByNameResult<'mir, 'tcx>> {
        let this = self.eval_context_mut();

        // Windows API stubs.
        // HANDLE = isize
        // DWORD = ULONG = u32
        // BOOL = i32
        // BOOLEAN = u8
        match &*link_name.as_str() {
            // Environment related shims
            "GetEnvironmentVariableW" => {
                let &[ref name, ref buf, ref size] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let result = this.GetEnvironmentVariableW(name, buf, size)?;
                this.write_scalar(Scalar::from_u32(result), dest)?;
            }
            "SetEnvironmentVariableW" => {
                let &[ref name, ref value] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let result = this.SetEnvironmentVariableW(name, value)?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "GetEnvironmentStringsW" => {
                let &[] = this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let result = this.GetEnvironmentStringsW()?;
                this.write_pointer(result, dest)?;
            }
            "FreeEnvironmentStringsW" => {
                let &[ref env_block] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let result = this.FreeEnvironmentStringsW(env_block)?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "GetCurrentDirectoryW" => {
                let &[ref size, ref buf] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let result = this.GetCurrentDirectoryW(size, buf)?;
                this.write_scalar(Scalar::from_u32(result), dest)?;
            }
            "SetCurrentDirectoryW" => {
                let &[ref path] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let result = this.SetCurrentDirectoryW(path)?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }

            // File related shims
            "GetStdHandle" => {
                let &[ref which] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let which = this.read_scalar(which)?.to_i32()?;
                // We just make this the identity function, so we know later in `WriteFile`
                // which one it is.
                this.write_scalar(Scalar::from_machine_isize(which.into(), this), dest)?;
            }
            "WriteFile" => {
                let &[ref handle, ref buf, ref n, ref written_ptr, ref overlapped] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.read_scalar(overlapped)?.to_machine_usize(this)?; // this is a poiner, that we ignore
                let handle = this.read_scalar(handle)?.to_machine_isize(this)?;
                let buf = this.read_pointer(buf)?;
                let n = this.read_scalar(n)?.to_u32()?;
                let written_place = this.deref_operand(written_ptr)?;
                // Spec says to always write `0` first.
                this.write_null(&written_place.into())?;
                let written = if handle == -11 || handle == -12 {
                    // stdout/stderr
                    use std::io::{self, Write};

                    let buf_cont = this.memory.read_bytes(buf, Size::from_bytes(u64::from(n)))?;
                    let res = if handle == -11 {
                        io::stdout().write(buf_cont)
                    } else {
                        io::stderr().write(buf_cont)
                    };
                    res.ok().map(|n| n as u32)
                } else {
                    throw_unsup_format!(
                        "on Windows, writing to anything except stdout/stderr is not supported"
                    )
                };
                // If there was no error, write back how much was written.
                if let Some(n) = written {
                    this.write_scalar(Scalar::from_u32(n), &written_place.into())?;
                }
                // Return whether this was a success.
                this.write_scalar(Scalar::from_i32(if written.is_some() { 1 } else { 0 }), dest)?;
            }

            // Allocation
            "HeapAlloc" => {
                let &[ref handle, ref flags, ref size] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.read_scalar(handle)?.to_machine_isize(this)?;
                let flags = this.read_scalar(flags)?.to_u32()?;
                let size = this.read_scalar(size)?.to_machine_usize(this)?;
                let zero_init = (flags & 0x00000008) != 0; // HEAP_ZERO_MEMORY
                let res = this.malloc(size, zero_init, MiriMemoryKind::WinHeap)?;
                this.write_pointer(res, dest)?;
            }
            "HeapFree" => {
                let &[ref handle, ref flags, ref ptr] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.read_scalar(handle)?.to_machine_isize(this)?;
                this.read_scalar(flags)?.to_u32()?;
                let ptr = this.read_pointer(ptr)?;
                this.free(ptr, MiriMemoryKind::WinHeap)?;
                this.write_scalar(Scalar::from_i32(1), dest)?;
            }
            "HeapReAlloc" => {
                let &[ref handle, ref flags, ref ptr, ref size] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.read_scalar(handle)?.to_machine_isize(this)?;
                this.read_scalar(flags)?.to_u32()?;
                let ptr = this.read_pointer(ptr)?;
                let size = this.read_scalar(size)?.to_machine_usize(this)?;
                let res = this.realloc(ptr, size, MiriMemoryKind::WinHeap)?;
                this.write_pointer(res, dest)?;
            }

            // errno
            "SetLastError" => {
                let &[ref error] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let error = this.read_scalar(error)?.check_init()?;
                this.set_last_error(error)?;
            }
            "GetLastError" => {
                let &[] = this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let last_error = this.get_last_error()?;
                this.write_scalar(last_error, dest)?;
            }

            // Querying system information
            "GetSystemInfo" => {
                let &[ref system_info] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let system_info = this.deref_operand(system_info)?;
                // Initialize with `0`.
                this.memory.write_bytes(
                    system_info.ptr,
                    iter::repeat(0u8).take(system_info.layout.size.bytes() as usize),
                )?;
                // Set number of processors.
                let dword_size = Size::from_bytes(4);
                let num_cpus = this.mplace_field(&system_info, 6)?;
                this.write_scalar(Scalar::from_int(NUM_CPUS, dword_size), &num_cpus.into())?;
            }

            // Thread-local storage
            "TlsAlloc" => {
                // This just creates a key; Windows does not natively support TLS destructors.

                // Create key and return it.
                let &[] = this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let key = this.machine.tls.create_tls_key(None, dest.layout.size)?;
                this.write_scalar(Scalar::from_uint(key, dest.layout.size), dest)?;
            }
            "TlsGetValue" => {
                let &[ref key] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let key = u128::from(this.read_scalar(key)?.to_u32()?);
                let active_thread = this.get_active_thread();
                let ptr = this.machine.tls.load_tls(key, active_thread, this)?;
                this.write_scalar(ptr, dest)?;
            }
            "TlsSetValue" => {
                let &[ref key, ref new_ptr] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let key = u128::from(this.read_scalar(key)?.to_u32()?);
                let active_thread = this.get_active_thread();
                let new_data = this.read_scalar(new_ptr)?.check_init()?;
                this.machine.tls.store_tls(key, active_thread, new_data, &*this.tcx)?;

                // Return success (`1`).
                this.write_scalar(Scalar::from_i32(1), dest)?;
            }

            // Access to command-line arguments
            "GetCommandLineW" => {
                let &[] = this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.write_pointer(
                    this.machine.cmd_line.expect("machine must be initialized").ptr,
                    dest,
                )?;
            }

            // Time related shims
            "GetSystemTimeAsFileTime" => {
                #[allow(non_snake_case)]
                let &[ref LPFILETIME] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.GetSystemTimeAsFileTime(LPFILETIME)?;
            }
            "QueryPerformanceCounter" => {
                #[allow(non_snake_case)]
                let &[ref lpPerformanceCount] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let result = this.QueryPerformanceCounter(lpPerformanceCount)?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "QueryPerformanceFrequency" => {
                #[allow(non_snake_case)]
                let &[ref lpFrequency] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let result = this.QueryPerformanceFrequency(lpFrequency)?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }

            // Synchronization primitives
            "AcquireSRWLockExclusive" => {
                let &[ref ptr] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.AcquireSRWLockExclusive(ptr)?;
            }
            "ReleaseSRWLockExclusive" => {
                let &[ref ptr] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.ReleaseSRWLockExclusive(ptr)?;
            }
            "TryAcquireSRWLockExclusive" => {
                let &[ref ptr] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let ret = this.TryAcquireSRWLockExclusive(ptr)?;
                this.write_scalar(Scalar::from_u8(ret), dest)?;
            }
            "AcquireSRWLockShared" => {
                let &[ref ptr] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.AcquireSRWLockShared(ptr)?;
            }
            "ReleaseSRWLockShared" => {
                let &[ref ptr] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.ReleaseSRWLockShared(ptr)?;
            }
            "TryAcquireSRWLockShared" => {
                let &[ref ptr] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let ret = this.TryAcquireSRWLockShared(ptr)?;
                this.write_scalar(Scalar::from_u8(ret), dest)?;
            }

            // Dynamic symbol loading
            "GetProcAddress" => {
                #[allow(non_snake_case)]
                let &[ref hModule, ref lpProcName] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.read_scalar(hModule)?.to_machine_isize(this)?;
                let name = this.read_c_str(this.read_pointer(lpProcName)?)?;
                if let Some(dlsym) = Dlsym::from_str(name, &this.tcx.sess.target.os)? {
                    let ptr = this.memory.create_fn_alloc(FnVal::Other(dlsym));
                    this.write_pointer(ptr, dest)?;
                } else {
                    this.write_null(dest)?;
                }
            }

            // Miscellaneous
            "SystemFunction036" => {
                // This is really 'RtlGenRandom'.
                let &[ref ptr, ref len] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                let len = this.read_scalar(len)?.to_u32()?;
                this.gen_random(ptr, len.into())?;
                this.write_scalar(Scalar::from_bool(true), dest)?;
            }
            "BCryptGenRandom" => {
                let &[ref algorithm, ref ptr, ref len, ref flags] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let algorithm = this.read_scalar(algorithm)?;
                let ptr = this.read_pointer(ptr)?;
                let len = this.read_scalar(len)?.to_u32()?;
                let flags = this.read_scalar(flags)?.to_u32()?;
                if flags != 2 {
                    //      ^ BCRYPT_USE_SYSTEM_PREFERRED_RNG
                    throw_unsup_format!(
                        "BCryptGenRandom is supported only with the BCRYPT_USE_SYSTEM_PREFERRED_RNG flag"
                    );
                }
                if algorithm.to_machine_usize(this)? != 0 {
                    throw_unsup_format!(
                        "BCryptGenRandom algorithm must be NULL when the flag is BCRYPT_USE_SYSTEM_PREFERRED_RNG"
                    );
                }
                this.gen_random(ptr, len.into())?;
                this.write_null(dest)?; // STATUS_SUCCESS
            }
            "GetConsoleScreenBufferInfo" => {
                // `term` needs this, so we fake it.
                let &[ref console, ref buffer_info] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.read_scalar(console)?.to_machine_isize(this)?;
                this.deref_operand(buffer_info)?;
                // Indicate an error.
                // FIXME: we should set last_error, but to what?
                this.write_null(dest)?;
            }
            "GetConsoleMode" => {
                // Windows "isatty" (in libtest) needs this, so we fake it.
                let &[ref console, ref mode] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.read_scalar(console)?.to_machine_isize(this)?;
                this.deref_operand(mode)?;
                // Indicate an error.
                // FIXME: we should set last_error, but to what?
                this.write_null(dest)?;
            }
            "SwitchToThread" => {
                let &[] = this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                // Note that once Miri supports concurrency, this will need to return a nonzero
                // value if this call does result in switching to another thread.
                this.write_null(dest)?;
            }

            // Better error for attempts to create a thread
            "CreateThread" => {
                let &[_, _, _, _, _, _] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;

                this.handle_unsupported("can't create threads on Windows")?;
                return Ok(EmulateByNameResult::AlreadyJumped);
            }

            // Incomplete shims that we "stub out" just to get pre-main initialization code to work.
            // These shims are enabled only when the caller is in the standard library.
            "GetProcessHeap" if this.frame_in_std() => {
                let &[] = this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                // Just fake a HANDLE
                this.write_scalar(Scalar::from_machine_isize(1, this), dest)?;
            }
            "SetConsoleTextAttribute" if this.frame_in_std() => {
                #[allow(non_snake_case)]
                let &[ref _hConsoleOutput, ref _wAttribute] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                // Pretend these does not exist / nothing happened, by returning zero.
                this.write_null(dest)?;
            }
            "AddVectoredExceptionHandler" if this.frame_in_std() => {
                #[allow(non_snake_case)]
                let &[ref _First, ref _Handler] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                // Any non zero value works for the stdlib. This is just used for stack overflows anyway.
                this.write_scalar(Scalar::from_machine_usize(1, this), dest)?;
            }
            "SetThreadStackGuarantee" if this.frame_in_std() => {
                #[allow(non_snake_case)]
                let &[_StackSizeInBytes] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                // Any non zero value works for the stdlib. This is just used for stack overflows anyway.
                this.write_scalar(Scalar::from_u32(1), dest)?;
            }
            | "InitializeCriticalSection"
            | "EnterCriticalSection"
            | "LeaveCriticalSection"
            | "DeleteCriticalSection"
                if this.frame_in_std() =>
            {
                #[allow(non_snake_case)]
                let &[ref _lpCriticalSection] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                assert_eq!(
                    this.get_total_thread_count(),
                    1,
                    "concurrency on Windows is not supported"
                );
                // Nothing to do, not even a return value.
                // (Windows locks are reentrant, and we have only 1 thread,
                // so not doing any futher checks here is at least not incorrect.)
            }
            "TryEnterCriticalSection" if this.frame_in_std() => {
                #[allow(non_snake_case)]
                let &[ref _lpCriticalSection] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                assert_eq!(
                    this.get_total_thread_count(),
                    1,
                    "concurrency on Windows is not supported"
                );
                // There is only one thread, so this always succeeds and returns TRUE.
                this.write_scalar(Scalar::from_i32(1), dest)?;
            }

            _ => return Ok(EmulateByNameResult::NotSupported),
        }

        Ok(EmulateByNameResult::NeedsJumping)
    }
}
