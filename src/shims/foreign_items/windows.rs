use std::iter;

use rustc_middle::mir;
use rustc_target::abi::Size;

use crate::*;

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn emulate_foreign_item_by_name(
        &mut self,
        link_name: &str,
        args: &[OpTy<'tcx, Tag>],
        dest: PlaceTy<'tcx, Tag>,
        _ret: mir::BasicBlock,
    ) -> InterpResult<'tcx, bool> {
        let this = self.eval_context_mut();

        // Windows API stubs.
        // HANDLE = isize
        // DWORD = ULONG = u32
        // BOOL = i32
        match link_name {
            // Environment related shims
            "GetEnvironmentVariableW" => {
                let result = this.GetEnvironmentVariableW(args[0], args[1], args[2])?;
                this.write_scalar(Scalar::from_u32(result), dest)?;
            }
            "SetEnvironmentVariableW" => {
                let result = this.SetEnvironmentVariableW(args[0], args[1])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "GetEnvironmentStringsW" => {
                let result = this.GetEnvironmentStringsW()?;
                this.write_scalar(result, dest)?;
            }
            "FreeEnvironmentStringsW" => {
                let result = this.FreeEnvironmentStringsW(args[0])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "GetCurrentDirectoryW" => {
                let result = this.GetCurrentDirectoryW(args[0], args[1])?;
                this.write_scalar(Scalar::from_u32(result), dest)?;
            }
            "SetCurrentDirectoryW" => {
                let result = this.SetCurrentDirectoryW(args[0])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }

            // File related shims
            "GetStdHandle" => {
                let which = this.read_scalar(args[0])?.to_i32()?;
                // We just make this the identity function, so we know later in `WriteFile`
                // which one it is.
                this.write_scalar(Scalar::from_machine_isize(which.into(), this), dest)?;
            }
            "WriteFile" => {
                let handle = this.read_scalar(args[0])?.to_machine_isize(this)?;
                let buf = this.read_scalar(args[1])?.not_undef()?;
                let n = this.read_scalar(args[2])?.to_u32()?;
                let written_place = this.deref_operand(args[3])?;
                // Spec says to always write `0` first.
                this.write_null(written_place.into())?;
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
                    this.write_scalar(Scalar::from_u32(n), written_place.into())?;
                }
                // Return whether this was a success.
                this.write_scalar(
                    Scalar::from_i32(if written.is_some() { 1 } else { 0 }),
                    dest,
                )?;
            }

            // Allocation
            "HeapAlloc" => {
                let _handle = this.read_scalar(args[0])?.to_machine_isize(this)?;
                let flags = this.read_scalar(args[1])?.to_u32()?;
                let size = this.read_scalar(args[2])?.to_machine_usize(this)?;
                let zero_init = (flags & 0x00000008) != 0; // HEAP_ZERO_MEMORY
                let res = this.malloc(size, zero_init, MiriMemoryKind::WinHeap);
                this.write_scalar(res, dest)?;
            }
            "HeapFree" => {
                let _handle = this.read_scalar(args[0])?.to_machine_isize(this)?;
                let _flags = this.read_scalar(args[1])?.to_u32()?;
                let ptr = this.read_scalar(args[2])?.not_undef()?;
                this.free(ptr, MiriMemoryKind::WinHeap)?;
                this.write_scalar(Scalar::from_i32(1), dest)?;
            }
            "HeapReAlloc" => {
                let _handle = this.read_scalar(args[0])?.to_machine_isize(this)?;
                let _flags = this.read_scalar(args[1])?.to_u32()?;
                let ptr = this.read_scalar(args[2])?.not_undef()?;
                let size = this.read_scalar(args[3])?.to_machine_usize(this)?;
                let res = this.realloc(ptr, size, MiriMemoryKind::WinHeap)?;
                this.write_scalar(res, dest)?;
            }

            // errno
            "SetLastError" => {
                this.set_last_error(this.read_scalar(args[0])?.not_undef()?)?;
            }
            "GetLastError" => {
                let last_error = this.get_last_error()?;
                this.write_scalar(last_error, dest)?;
            }

            // Querying system information
            "GetSystemInfo" => {
                let system_info = this.deref_operand(args[0])?;
                // Initialize with `0`.
                this.memory.write_bytes(
                    system_info.ptr,
                    iter::repeat(0u8).take(system_info.layout.size.bytes() as usize),
                )?;
                // Set number of processors.
                let dword_size = Size::from_bytes(4);
                let num_cpus = this.mplace_field(system_info, 6)?;
                this.write_scalar(Scalar::from_int(NUM_CPUS, dword_size), num_cpus.into())?;
            }

            // Thread-local storage
            "TlsAlloc" => {
                // This just creates a key; Windows does not natively support TLS destructors.

                // Create key and return it.
                let key = this.machine.tls.create_tls_key(None, dest.layout.size)?;
                this.write_scalar(Scalar::from_uint(key, dest.layout.size), dest)?;
            }
            "TlsGetValue" => {
                let key = u128::from(this.read_scalar(args[0])?.to_u32()?);
                let ptr = this.machine.tls.load_tls(key, this)?;
                this.write_scalar(ptr, dest)?;
            }
            "TlsSetValue" => {
                let key = u128::from(this.read_scalar(args[0])?.to_u32()?);
                let new_ptr = this.read_scalar(args[1])?.not_undef()?;
                this.machine.tls.store_tls(key, this.test_null(new_ptr)?)?;

                // Return success (`1`).
                this.write_scalar(Scalar::from_i32(1), dest)?;
            }

            // Access to command-line arguments
            "GetCommandLineW" => {
                this.write_scalar(
                    this.machine.cmd_line.expect("machine must be initialized"),
                    dest,
                )?;
            }

            // Time related shims
            "GetSystemTimeAsFileTime" => {
                this.GetSystemTimeAsFileTime(args[0])?;
            }
            "QueryPerformanceCounter" => {
                let result = this.QueryPerformanceCounter(args[0])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "QueryPerformanceFrequency" => {
                let result = this.QueryPerformanceFrequency(args[0])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }

            // Miscellaneous
            "SystemFunction036" => {
                // The actual name of 'RtlGenRandom'
                let ptr = this.read_scalar(args[0])?.not_undef()?;
                let len = this.read_scalar(args[1])?.to_u32()?;
                this.gen_random(ptr, len.into())?;
                this.write_scalar(Scalar::from_bool(true), dest)?;
            }
            "GetConsoleScreenBufferInfo" => {
                // `term` needs this, so we fake it.
                let _console = this.read_scalar(args[0])?.to_machine_isize(this)?;
                let _buffer_info = this.deref_operand(args[1])?;
                // Indicate an error.
                // FIXME: we should set last_error, but to what?
                this.write_null(dest)?;
            }
            "GetConsoleMode" => {
                // Windows "isatty" (in libtest) needs this, so we fake it.
                let _console = this.read_scalar(args[0])?.to_machine_isize(this)?;
                let _mode = this.deref_operand(args[1])?;
                // Indicate an error.
                // FIXME: we should set last_error, but to what?
                this.write_null(dest)?;
            }

            // Better error for attempts to create a thread
            "CreateThread" => {
                throw_unsup_format!("Miri does not support threading");
            }

            // Incomplete shims that we "stub out" just to get pre-main initialization code to work.
            // These shims are enabled only when the caller is in the standard library.
            "GetProcessHeap" if this.frame().instance.to_string().starts_with("std::sys::windows::") => {
                // Just fake a HANDLE
                this.write_scalar(Scalar::from_machine_isize(1, this), dest)?;
            }
            | "GetModuleHandleW"
            | "GetProcAddress"
            | "SetConsoleTextAttribute" if this.frame().instance.to_string().starts_with("std::sys::windows::")
            => {
                // Pretend these do not exist / nothing happened, by returning zero.
                this.write_null(dest)?;
            }
            "AddVectoredExceptionHandler" if this.frame().instance.to_string().starts_with("std::sys::windows::") => {
                // Any non zero value works for the stdlib. This is just used for stack overflows anyway.
                this.write_scalar(Scalar::from_machine_usize(1, this), dest)?;
            }
            | "InitializeCriticalSection"
            | "EnterCriticalSection"
            | "LeaveCriticalSection"
            | "DeleteCriticalSection" if this.frame().instance.to_string().starts_with("std::sys::windows::")
            => {
                // Nothing to do, not even a return value.
                // (Windows locks are reentrant, and we have only 1 thread,
                // so not doing any futher checks here is at least not incorrect.)
            }
            "TryEnterCriticalSection" if this.frame().instance.to_string().starts_with("std::sys::windows::")
            => {
                // There is only one thread, so this always succeeds and returns TRUE
                this.write_scalar(Scalar::from_i32(1), dest)?;
            }

            _ => throw_unsup_format!("can't call foreign function: {}", link_name),
        }

        Ok(true)
    }
}

