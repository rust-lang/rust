use crate::*;
use rustc::mir;
use rustc::ty::layout::Size;
use std::iter;

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
        let tcx = &{ this.tcx.tcx };

        match link_name {
            // Windows API stubs.
            // HANDLE = isize
            // DWORD = ULONG = u32
            // BOOL = i32

            // Environment related shims
            "GetEnvironmentVariableW" => {
                // args[0] : LPCWSTR lpName (32-bit ptr to a const string of 16-bit Unicode chars)
                // args[1] : LPWSTR lpBuffer (32-bit pointer to a string of 16-bit Unicode chars)
                // lpBuffer : ptr to buffer that receives contents of the env_var as a null-terminated string.
                // Return `# of chars` stored in the buffer pointed to by lpBuffer, excluding null-terminator.
                // Return 0 upon failure.

                // This is not the env var you are looking for.
                this.set_last_error(Scalar::from_u32(203))?; // ERROR_ENVVAR_NOT_FOUND
                this.write_null(dest)?;
            }

            "SetEnvironmentVariableW" => {
                // args[0] : LPCWSTR lpName (32-bit ptr to a const string of 16-bit Unicode chars)
                // args[1] : LPCWSTR lpValue (32-bit ptr to a const string of 16-bit Unicode chars)
                // Return nonzero if success, else return 0.
                throw_unsup_format!("can't set environment variable on Windows");
            }

            // File related shims
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
                    eprintln!("Miri: Ignored output to handle {}", handle);
                    // Pretend it all went well.
                    Some(n)
                };
                // If there was no error, write back how much was written.
                if let Some(n) = written {
                    this.write_scalar(Scalar::from_u32(n), written_place.into())?;
                }
                // Return whether this was a success.
                this.write_scalar(
                    Scalar::from_int(if written.is_some() { 1 } else { 0 }, dest.layout.size),
                    dest,
                )?;
            }

            // Other shims
            "GetProcessHeap" => {
                // Just fake a HANDLE
                this.write_scalar(Scalar::from_int(1, this.pointer_size()), dest)?;
            }
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
                this.write_scalar(Scalar::from_int(1, Size::from_bytes(4)), dest)?;
            }
            "HeapReAlloc" => {
                let _handle = this.read_scalar(args[0])?.to_machine_isize(this)?;
                let _flags = this.read_scalar(args[1])?.to_u32()?;
                let ptr = this.read_scalar(args[2])?.not_undef()?;
                let size = this.read_scalar(args[3])?.to_machine_usize(this)?;
                let res = this.realloc(ptr, size, MiriMemoryKind::WinHeap)?;
                this.write_scalar(res, dest)?;
            }

            "SetLastError" => {
                this.set_last_error(this.read_scalar(args[0])?.not_undef()?)?;
            }
            "GetLastError" => {
                let last_error = this.get_last_error()?;
                this.write_scalar(last_error, dest)?;
            }

            "AddVectoredExceptionHandler" => {
                // Any non zero value works for the stdlib. This is just used for stack overflows anyway.
                this.write_scalar(Scalar::from_int(1, dest.layout.size), dest)?;
            }

            | "InitializeCriticalSection"
            | "EnterCriticalSection"
            | "LeaveCriticalSection"
            | "DeleteCriticalSection"
            => {
                // Nothing to do, not even a return value.
                // (Windows locks are reentrant, and we have only 1 thread,
                // so not doing any futher checks here is at least not incorrect.)
            }

            | "GetModuleHandleW"
            | "GetProcAddress"
            | "GetConsoleScreenBufferInfo"
            | "SetConsoleTextAttribute"
            => {
                // Pretend these do not exist / nothing happened, by returning zero.
                this.write_null(dest)?;
            }

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

            "TlsAlloc" => {
                // This just creates a key; Windows does not natively support TLS destructors.

                // Create key and return it.
                let key = this.machine.tls.create_tls_key(None) as u128;

                // Figure out how large a TLS key actually is. This is `c::DWORD`.
                if dest.layout.size.bits() < 128
                    && key >= (1u128 << dest.layout.size.bits() as u128)
                {
                    throw_unsup!(OutOfTls);
                }
                this.write_scalar(Scalar::from_uint(key, dest.layout.size), dest)?;
            }
            "TlsGetValue" => {
                let key = this.read_scalar(args[0])?.to_u32()? as u128;
                let ptr = this.machine.tls.load_tls(key, tcx)?;
                this.write_scalar(ptr, dest)?;
            }
            "TlsSetValue" => {
                let key = this.read_scalar(args[0])?.to_u32()? as u128;
                let new_ptr = this.read_scalar(args[1])?.not_undef()?;
                this.machine.tls.store_tls(key, this.test_null(new_ptr)?)?;

                // Return success (`1`).
                this.write_scalar(Scalar::from_int(1, dest.layout.size), dest)?;
            }
            "GetStdHandle" => {
                let which = this.read_scalar(args[0])?.to_i32()?;
                // We just make this the identity function, so we know later in `WriteFile`
                // which one it is.
                this.write_scalar(Scalar::from_int(which, this.pointer_size()), dest)?;
            }
            "GetConsoleMode" => {
                // Everything is a pipe.
                this.write_null(dest)?;
            }
            "GetCommandLineW" => {
                this.write_scalar(
                    this.machine.cmd_line.expect("machine must be initialized"),
                    dest,
                )?;
            }
            // The actual name of 'RtlGenRandom'
            "SystemFunction036" => {
                let ptr = this.read_scalar(args[0])?.not_undef()?;
                let len = this.read_scalar(args[1])?.to_u32()?;
                this.gen_random(ptr, len as usize)?;
                this.write_scalar(Scalar::from_bool(true), dest)?;
            }
            // We don't support threading.
            "CreateThread" => {
                throw_unsup_format!("Miri does not support threading");
            }
            _ => throw_unsup_format!("can't call foreign function: {}", link_name),
        }

        Ok(true)
    }
}

