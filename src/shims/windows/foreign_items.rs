use std::iter;

use rustc_middle::mir;
use rustc_target::abi::Size;
use rustc_target::spec::abi::Abi;

use crate::*;
use helpers::{check_abi, check_arg_count};
use shims::windows::sync::EvalContextExt as _;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn emulate_foreign_item_by_name(
        &mut self,
        link_name: &str,
        abi: Abi,
        args: &[OpTy<'tcx, Tag>],
        dest: &PlaceTy<'tcx, Tag>,
        _ret: mir::BasicBlock,
    ) -> InterpResult<'tcx, bool> {
        let this = self.eval_context_mut();

        // Windows API stubs.
        // HANDLE = isize
        // DWORD = ULONG = u32
        // BOOL = i32
        // BOOLEAN = u8
        match link_name {
            // Environment related shims
            "GetEnvironmentVariableW" => {
                check_abi(abi, Abi::System { unwind: false })?;
                let &[ref name, ref buf, ref size] = check_arg_count(args)?;
                let result = this.GetEnvironmentVariableW(name, buf, size)?;
                this.write_scalar(Scalar::from_u32(result), dest)?;
            }
            "SetEnvironmentVariableW" => {
                check_abi(abi, Abi::System { unwind: false })?;
                let &[ref name, ref value] = check_arg_count(args)?;
                let result = this.SetEnvironmentVariableW(name, value)?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "GetEnvironmentStringsW" => {
                check_abi(abi, Abi::System { unwind: false })?;
                let &[] = check_arg_count(args)?;
                let result = this.GetEnvironmentStringsW()?;
                this.write_scalar(result, dest)?;
            }
            "FreeEnvironmentStringsW" => {
                check_abi(abi, Abi::System { unwind: false })?;
                let &[ref env_block] = check_arg_count(args)?;
                let result = this.FreeEnvironmentStringsW(env_block)?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "GetCurrentDirectoryW" => {
                check_abi(abi, Abi::System { unwind: false })?;
                let &[ref size, ref buf] = check_arg_count(args)?;
                let result = this.GetCurrentDirectoryW(size, buf)?;
                this.write_scalar(Scalar::from_u32(result), dest)?;
            }
            "SetCurrentDirectoryW" => {
                check_abi(abi, Abi::System { unwind: false })?;
                let &[ref path] = check_arg_count(args)?;
                let result = this.SetCurrentDirectoryW(path)?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }

            // File related shims
            "GetStdHandle" => {
                check_abi(abi, Abi::System { unwind: false })?;
                let &[ref which] = check_arg_count(args)?;
                let which = this.read_scalar(which)?.to_i32()?;
                // We just make this the identity function, so we know later in `WriteFile`
                // which one it is.
                this.write_scalar(Scalar::from_machine_isize(which.into(), this), dest)?;
            }
            "WriteFile" => {
                check_abi(abi, Abi::System { unwind: false })?;
                let &[ref handle, ref buf, ref n, ref written_ptr, ref overlapped] = check_arg_count(args)?;
                this.read_scalar(overlapped)?.to_machine_usize(this)?; // this is a poiner, that we ignore
                let handle = this.read_scalar(handle)?.to_machine_isize(this)?;
                let buf = this.read_scalar(buf)?.check_init()?;
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
                    throw_unsup_format!("on Windows, writing to anything except stdout/stderr is not supported")
                };
                // If there was no error, write back how much was written.
                if let Some(n) = written {
                    this.write_scalar(Scalar::from_u32(n), &written_place.into())?;
                }
                // Return whether this was a success.
                this.write_scalar(
                    Scalar::from_i32(if written.is_some() { 1 } else { 0 }),
                    dest,
                )?;
            }

            // Allocation
            "HeapAlloc" => {
                check_abi(abi, Abi::System { unwind: false })?;
                let &[ref handle, ref flags, ref size] = check_arg_count(args)?;
                this.read_scalar(handle)?.to_machine_isize(this)?;
                let flags = this.read_scalar(flags)?.to_u32()?;
                let size = this.read_scalar(size)?.to_machine_usize(this)?;
                let zero_init = (flags & 0x00000008) != 0; // HEAP_ZERO_MEMORY
                let res = this.malloc(size, zero_init, MiriMemoryKind::WinHeap);
                this.write_scalar(res, dest)?;
            }
            "HeapFree" => {
                check_abi(abi, Abi::System { unwind: false })?;
                let &[ref handle, ref flags, ref ptr] = check_arg_count(args)?;
                this.read_scalar(handle)?.to_machine_isize(this)?;
                this.read_scalar(flags)?.to_u32()?;
                let ptr = this.read_scalar(ptr)?.check_init()?;
                this.free(ptr, MiriMemoryKind::WinHeap)?;
                this.write_scalar(Scalar::from_i32(1), dest)?;
            }
            "HeapReAlloc" => {
                check_abi(abi, Abi::System { unwind: false })?;
                let &[ref handle, ref flags, ref ptr, ref size] = check_arg_count(args)?;
                this.read_scalar(handle)?.to_machine_isize(this)?;
                this.read_scalar(flags)?.to_u32()?;
                let ptr = this.read_scalar(ptr)?.check_init()?;
                let size = this.read_scalar(size)?.to_machine_usize(this)?;
                let res = this.realloc(ptr, size, MiriMemoryKind::WinHeap)?;
                this.write_scalar(res, dest)?;
            }

            // errno
            "SetLastError" => {
                check_abi(abi, Abi::System { unwind: false })?;
                let &[ref error] = check_arg_count(args)?;
                let error = this.read_scalar(error)?.check_init()?;
                this.set_last_error(error)?;
            }
            "GetLastError" => {
                check_abi(abi, Abi::System { unwind: false })?;
                let &[] = check_arg_count(args)?;
                let last_error = this.get_last_error()?;
                this.write_scalar(last_error, dest)?;
            }

            // Querying system information
            "GetSystemInfo" => {
                check_abi(abi, Abi::System { unwind: false })?;
                let &[ref system_info] = check_arg_count(args)?;
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
                check_abi(abi, Abi::System { unwind: false })?;
                // This just creates a key; Windows does not natively support TLS destructors.

                // Create key and return it.
                let &[] = check_arg_count(args)?;
                let key = this.machine.tls.create_tls_key(None, dest.layout.size)?;
                this.write_scalar(Scalar::from_uint(key, dest.layout.size), dest)?;
            }
            "TlsGetValue" => {
                check_abi(abi, Abi::System { unwind: false })?;
                let &[ref key] = check_arg_count(args)?;
                let key = u128::from(this.read_scalar(key)?.to_u32()?);
                let active_thread = this.get_active_thread();
                let ptr = this.machine.tls.load_tls(key, active_thread, this)?;
                this.write_scalar(ptr, dest)?;
            }
            "TlsSetValue" => {
                check_abi(abi, Abi::System { unwind: false })?;
                let &[ref key, ref new_ptr] = check_arg_count(args)?;
                let key = u128::from(this.read_scalar(key)?.to_u32()?);
                let active_thread = this.get_active_thread();
                let new_ptr = this.read_scalar(new_ptr)?.check_init()?;
                this.machine.tls.store_tls(key, active_thread, this.test_null(new_ptr)?)?;

                // Return success (`1`).
                this.write_scalar(Scalar::from_i32(1), dest)?;
            }

            // Access to command-line arguments
            "GetCommandLineW" => {
                check_abi(abi, Abi::System { unwind: false })?;
                let &[] = check_arg_count(args)?;
                this.write_scalar(
                    this.machine.cmd_line.expect("machine must be initialized"),
                    dest,
                )?;
            }

            // Time related shims
            "GetSystemTimeAsFileTime" => {
                check_abi(abi, Abi::System { unwind: false })?;
                #[allow(non_snake_case)]
                let &[ref LPFILETIME] = check_arg_count(args)?;
                this.GetSystemTimeAsFileTime(LPFILETIME)?;
            }
            "QueryPerformanceCounter" => {
                check_abi(abi, Abi::System { unwind: false })?;
                #[allow(non_snake_case)]
                let &[ref lpPerformanceCount] = check_arg_count(args)?;
                let result = this.QueryPerformanceCounter(lpPerformanceCount)?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "QueryPerformanceFrequency" => {
                check_abi(abi, Abi::System { unwind: false })?;
                #[allow(non_snake_case)]
                let &[ref lpFrequency] = check_arg_count(args)?;
                let result = this.QueryPerformanceFrequency(lpFrequency)?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }

            // Synchronization primitives
            "AcquireSRWLockExclusive" => {
                check_abi(abi, Abi::System { unwind: false })?;
                let &[ref ptr] = check_arg_count(args)?;
                this.AcquireSRWLockExclusive(ptr)?;
            }
            "ReleaseSRWLockExclusive" => {
                check_abi(abi, Abi::System { unwind: false })?;
                let &[ref ptr] = check_arg_count(args)?;
                this.ReleaseSRWLockExclusive(ptr)?;
            }
            "TryAcquireSRWLockExclusive" => {
                check_abi(abi, Abi::System { unwind: false })?;
                let &[ref ptr] = check_arg_count(args)?;
                let ret = this.TryAcquireSRWLockExclusive(ptr)?;
                this.write_scalar(Scalar::from_u8(ret), dest)?;
            }
            "AcquireSRWLockShared" => {
                check_abi(abi, Abi::System { unwind: false })?;
                let &[ref ptr] = check_arg_count(args)?;
                this.AcquireSRWLockShared(ptr)?;
            }
            "ReleaseSRWLockShared" => {
                check_abi(abi, Abi::System { unwind: false })?;
                let &[ref ptr] = check_arg_count(args)?;
                this.ReleaseSRWLockShared(ptr)?;
            }
            "TryAcquireSRWLockShared" => {
                check_abi(abi, Abi::System { unwind: false })?;
                let &[ref ptr] = check_arg_count(args)?;
                let ret = this.TryAcquireSRWLockShared(ptr)?;
                this.write_scalar(Scalar::from_u8(ret), dest)?;
            }

            // Dynamic symbol loading
            "GetProcAddress" => {
                check_abi(abi, Abi::System { unwind: false })?;
                #[allow(non_snake_case)]
                let &[ref hModule, ref lpProcName] = check_arg_count(args)?;
                this.read_scalar(hModule)?.to_machine_isize(this)?;
                let name = this.memory.read_c_str(this.read_scalar(lpProcName)?.check_init()?)?;
                if let Some(dlsym) = Dlsym::from_str(name, &this.tcx.sess.target.os)? {
                    let ptr = this.memory.create_fn_alloc(FnVal::Other(dlsym));
                    this.write_scalar(Scalar::from(ptr), dest)?;
                } else {
                    this.write_null(dest)?;
                }
            }

            // Miscellaneous
            "SystemFunction036" => {
                check_abi(abi, Abi::System { unwind: false })?;
                // The actual name of 'RtlGenRandom'
                let &[ref ptr, ref len] = check_arg_count(args)?;
                let ptr = this.read_scalar(ptr)?.check_init()?;
                let len = this.read_scalar(len)?.to_u32()?;
                this.gen_random(ptr, len.into())?;
                this.write_scalar(Scalar::from_bool(true), dest)?;
            }
            "GetConsoleScreenBufferInfo" => {
                check_abi(abi, Abi::System { unwind: false })?;
                // `term` needs this, so we fake it.
                let &[ref console, ref buffer_info] = check_arg_count(args)?;
                this.read_scalar(console)?.to_machine_isize(this)?;
                this.deref_operand(buffer_info)?;
                // Indicate an error.
                // FIXME: we should set last_error, but to what?
                this.write_null(dest)?;
            }
            "GetConsoleMode" => {
                check_abi(abi, Abi::System { unwind: false })?;
                // Windows "isatty" (in libtest) needs this, so we fake it.
                let &[ref console, ref mode] = check_arg_count(args)?;
                this.read_scalar(console)?.to_machine_isize(this)?;
                this.deref_operand(mode)?;
                // Indicate an error.
                // FIXME: we should set last_error, but to what?
                this.write_null(dest)?;
            }
            "SwitchToThread" => {
                check_abi(abi, Abi::System { unwind: false })?;
                let &[] = check_arg_count(args)?;
                // Note that once Miri supports concurrency, this will need to return a nonzero
                // value if this call does result in switching to another thread.
                this.write_null(dest)?;
            }

            // Better error for attempts to create a thread
            "CreateThread" => {
                check_abi(abi, Abi::System { unwind: false })?;
                throw_unsup_format!("Miri does not support concurrency on Windows");
            }

            // Incomplete shims that we "stub out" just to get pre-main initialization code to work.
            // These shims are enabled only when the caller is in the standard library.
            "GetProcessHeap" if this.frame().instance.to_string().starts_with("std::sys::windows::") => {
                check_abi(abi, Abi::System { unwind: false })?;
                let &[] = check_arg_count(args)?;
                // Just fake a HANDLE
                this.write_scalar(Scalar::from_machine_isize(1, this), dest)?;
            }
            "SetConsoleTextAttribute" if this.frame().instance.to_string().starts_with("std::sys::windows::") => {
                check_abi(abi, Abi::System { unwind: false })?;
                #[allow(non_snake_case)]
                let &[ref _hConsoleOutput, ref _wAttribute] = check_arg_count(args)?;
                // Pretend these does not exist / nothing happened, by returning zero.
                this.write_null(dest)?;
            }
            "AddVectoredExceptionHandler" if this.frame().instance.to_string().starts_with("std::sys::windows::") => {
                check_abi(abi, Abi::System { unwind: false })?;
                #[allow(non_snake_case)]
                let &[ref _First, ref _Handler] = check_arg_count(args)?;
                // Any non zero value works for the stdlib. This is just used for stack overflows anyway.
                this.write_scalar(Scalar::from_machine_usize(1, this), dest)?;
            }
            "SetThreadStackGuarantee" if this.frame().instance.to_string().starts_with("std::sys::windows::") => {
                check_abi(abi, Abi::System { unwind: false })?;
                #[allow(non_snake_case)]
                let &[_StackSizeInBytes] = check_arg_count(args)?;
                // Any non zero value works for the stdlib. This is just used for stack overflows anyway.
                this.write_scalar(Scalar::from_u32(1), dest)?;
            }
            | "InitializeCriticalSection"
            | "EnterCriticalSection"
            | "LeaveCriticalSection"
            | "DeleteCriticalSection"
            if this.frame().instance.to_string().starts_with("std::sys::windows::") => {
                check_abi(abi, Abi::System { unwind: false })?;
                #[allow(non_snake_case)]
                let &[ref _lpCriticalSection] = check_arg_count(args)?;
                assert_eq!(this.get_total_thread_count(), 1, "concurrency on Windows is not supported");
                // Nothing to do, not even a return value.
                // (Windows locks are reentrant, and we have only 1 thread,
                // so not doing any futher checks here is at least not incorrect.)
            }
            "TryEnterCriticalSection"
            if this.frame().instance.to_string().starts_with("std::sys::windows::") => {
                check_abi(abi, Abi::System { unwind: false })?;
                #[allow(non_snake_case)]
                let &[ref _lpCriticalSection] = check_arg_count(args)?;
                assert_eq!(this.get_total_thread_count(), 1, "concurrency on Windows is not supported");
                // There is only one thread, so this always succeeds and returns TRUE.
                this.write_scalar(Scalar::from_i32(1), dest)?;
            }

            _ => throw_unsup_format!("can't call foreign function: {}", link_name),
        }

        Ok(true)
    }
}

