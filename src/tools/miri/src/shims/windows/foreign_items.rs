use std::iter;

use rustc_span::Symbol;
use rustc_target::abi::Size;
use rustc_target::spec::abi::Abi;

use crate::*;
use shims::foreign_items::EmulateByNameResult;
use shims::windows::handle::{EvalContextExt as _, Handle, PseudoHandle};
use shims::windows::sync::EvalContextExt as _;
use shims::windows::thread::EvalContextExt as _;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn emulate_foreign_item_by_name(
        &mut self,
        link_name: Symbol,
        abi: Abi,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, EmulateByNameResult<'mir, 'tcx>> {
        let this = self.eval_context_mut();

        // See `fn emulate_foreign_item_by_name` in `shims/foreign_items.rs` for the general pattern.

        // Windows API stubs.
        // HANDLE = isize
        // NTSTATUS = LONH = i32
        // DWORD = ULONG = u32
        // BOOL = i32
        // BOOLEAN = u8
        match link_name.as_str() {
            // Environment related shims
            "GetEnvironmentVariableW" => {
                let [name, buf, size] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let result = this.GetEnvironmentVariableW(name, buf, size)?;
                this.write_scalar(result, dest)?;
            }
            "SetEnvironmentVariableW" => {
                let [name, value] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let result = this.SetEnvironmentVariableW(name, value)?;
                this.write_scalar(result, dest)?;
            }
            "GetEnvironmentStringsW" => {
                let [] = this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let result = this.GetEnvironmentStringsW()?;
                this.write_pointer(result, dest)?;
            }
            "FreeEnvironmentStringsW" => {
                let [env_block] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let result = this.FreeEnvironmentStringsW(env_block)?;
                this.write_scalar(result, dest)?;
            }
            "GetCurrentDirectoryW" => {
                let [size, buf] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let result = this.GetCurrentDirectoryW(size, buf)?;
                this.write_scalar(result, dest)?;
            }
            "SetCurrentDirectoryW" => {
                let [path] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let result = this.SetCurrentDirectoryW(path)?;
                this.write_scalar(result, dest)?;
            }

            // File related shims
            "NtWriteFile" => {
                if !this.frame_in_std() {
                    throw_unsup_format!(
                        "`NtWriteFile` support is crude and just enough for stdout to work"
                    );
                }

                let [
                    handle,
                    _event,
                    _apc_routine,
                    _apc_context,
                    io_status_block,
                    buf,
                    n,
                    byte_offset,
                    _key,
                ] = this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let handle = this.read_target_isize(handle)?;
                let buf = this.read_pointer(buf)?;
                let n = this.read_scalar(n)?.to_u32()?;
                let byte_offset = this.read_target_usize(byte_offset)?; // is actually a pointer
                let io_status_block = this
                    .deref_operand_as(io_status_block, this.windows_ty_layout("IO_STATUS_BLOCK"))?;

                if byte_offset != 0 {
                    throw_unsup_format!(
                        "`NtWriteFile` `ByteOffset` parameter is non-null, which is unsupported"
                    );
                }

                let written = if handle == -11 || handle == -12 {
                    // stdout/stderr
                    use std::io::{self, Write};

                    let buf_cont =
                        this.read_bytes_ptr_strip_provenance(buf, Size::from_bytes(u64::from(n)))?;
                    let res = if this.machine.mute_stdout_stderr {
                        Ok(buf_cont.len())
                    } else if handle == -11 {
                        io::stdout().write(buf_cont)
                    } else {
                        io::stderr().write(buf_cont)
                    };
                    // We write at most `n` bytes, which is a `u32`, so we cannot have written more than that.
                    res.ok().map(|n| u32::try_from(n).unwrap())
                } else {
                    throw_unsup_format!(
                        "on Windows, writing to anything except stdout/stderr is not supported"
                    )
                };
                // We have to put the result into io_status_block.
                if let Some(n) = written {
                    let io_status_information =
                        this.project_field_named(&io_status_block, "Information")?;
                    this.write_scalar(
                        Scalar::from_target_usize(n.into(), this),
                        &io_status_information.into(),
                    )?;
                }
                // Return whether this was a success. >= 0 is success.
                // For the error code we arbitrarily pick 0xC0000185, STATUS_IO_DEVICE_ERROR.
                this.write_scalar(
                    Scalar::from_u32(if written.is_some() { 0 } else { 0xC0000185u32 }),
                    dest,
                )?;
            }

            // Allocation
            "HeapAlloc" => {
                let [handle, flags, size] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.read_target_isize(handle)?;
                let flags = this.read_scalar(flags)?.to_u32()?;
                let size = this.read_target_usize(size)?;
                let heap_zero_memory = 0x00000008; // HEAP_ZERO_MEMORY
                let zero_init = (flags & heap_zero_memory) == heap_zero_memory;
                let res = this.malloc(size, zero_init, MiriMemoryKind::WinHeap)?;
                this.write_pointer(res, dest)?;
            }
            "HeapFree" => {
                let [handle, flags, ptr] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.read_target_isize(handle)?;
                this.read_scalar(flags)?.to_u32()?;
                let ptr = this.read_pointer(ptr)?;
                this.free(ptr, MiriMemoryKind::WinHeap)?;
                this.write_scalar(Scalar::from_i32(1), dest)?;
            }
            "HeapReAlloc" => {
                let [handle, flags, ptr, size] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.read_target_isize(handle)?;
                this.read_scalar(flags)?.to_u32()?;
                let ptr = this.read_pointer(ptr)?;
                let size = this.read_target_usize(size)?;
                let res = this.realloc(ptr, size, MiriMemoryKind::WinHeap)?;
                this.write_pointer(res, dest)?;
            }

            // errno
            "SetLastError" => {
                let [error] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let error = this.read_scalar(error)?;
                this.set_last_error(error)?;
            }
            "GetLastError" => {
                let [] = this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let last_error = this.get_last_error()?;
                this.write_scalar(last_error, dest)?;
            }

            // Querying system information
            "GetSystemInfo" => {
                // Also called from `page_size` crate.
                let [system_info] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let system_info =
                    this.deref_operand_as(system_info, this.windows_ty_layout("SYSTEM_INFO"))?;
                // Initialize with `0`.
                this.write_bytes_ptr(
                    system_info.ptr,
                    iter::repeat(0u8).take(system_info.layout.size.bytes_usize()),
                )?;
                // Set selected fields.
                this.write_int_fields_named(
                    &[
                        ("dwPageSize", this.machine.page_size.into()),
                        ("dwNumberOfProcessors", this.machine.num_cpus.into()),
                    ],
                    &system_info,
                )?;
            }

            // Thread-local storage
            "TlsAlloc" => {
                // This just creates a key; Windows does not natively support TLS destructors.

                // Create key and return it.
                let [] = this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let key = this.machine.tls.create_tls_key(None, dest.layout.size)?;
                this.write_scalar(Scalar::from_uint(key, dest.layout.size), dest)?;
            }
            "TlsGetValue" => {
                let [key] = this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let key = u128::from(this.read_scalar(key)?.to_u32()?);
                let active_thread = this.get_active_thread();
                let ptr = this.machine.tls.load_tls(key, active_thread, this)?;
                this.write_scalar(ptr, dest)?;
            }
            "TlsSetValue" => {
                let [key, new_ptr] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let key = u128::from(this.read_scalar(key)?.to_u32()?);
                let active_thread = this.get_active_thread();
                let new_data = this.read_scalar(new_ptr)?;
                this.machine.tls.store_tls(key, active_thread, new_data, &*this.tcx)?;

                // Return success (`1`).
                this.write_scalar(Scalar::from_i32(1), dest)?;
            }

            // Access to command-line arguments
            "GetCommandLineW" => {
                let [] = this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.write_pointer(
                    this.machine.cmd_line.expect("machine must be initialized").ptr,
                    dest,
                )?;
            }

            // Time related shims
            "GetSystemTimeAsFileTime" => {
                #[allow(non_snake_case)]
                let [LPFILETIME] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.GetSystemTimeAsFileTime(LPFILETIME)?;
            }
            "QueryPerformanceCounter" => {
                #[allow(non_snake_case)]
                let [lpPerformanceCount] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let result = this.QueryPerformanceCounter(lpPerformanceCount)?;
                this.write_scalar(result, dest)?;
            }
            "QueryPerformanceFrequency" => {
                #[allow(non_snake_case)]
                let [lpFrequency] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let result = this.QueryPerformanceFrequency(lpFrequency)?;
                this.write_scalar(result, dest)?;
            }
            "Sleep" => {
                let [timeout] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;

                this.Sleep(timeout)?;
            }

            // Synchronization primitives
            "AcquireSRWLockExclusive" => {
                let [ptr] = this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.AcquireSRWLockExclusive(ptr)?;
            }
            "ReleaseSRWLockExclusive" => {
                let [ptr] = this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.ReleaseSRWLockExclusive(ptr)?;
            }
            "TryAcquireSRWLockExclusive" => {
                let [ptr] = this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let ret = this.TryAcquireSRWLockExclusive(ptr)?;
                this.write_scalar(ret, dest)?;
            }
            "AcquireSRWLockShared" => {
                let [ptr] = this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.AcquireSRWLockShared(ptr)?;
            }
            "ReleaseSRWLockShared" => {
                let [ptr] = this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.ReleaseSRWLockShared(ptr)?;
            }
            "TryAcquireSRWLockShared" => {
                let [ptr] = this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let ret = this.TryAcquireSRWLockShared(ptr)?;
                this.write_scalar(ret, dest)?;
            }
            "InitOnceBeginInitialize" => {
                let [ptr, flags, pending, context] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let result = this.InitOnceBeginInitialize(ptr, flags, pending, context)?;
                this.write_scalar(result, dest)?;
            }
            "InitOnceComplete" => {
                let [ptr, flags, context] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let result = this.InitOnceComplete(ptr, flags, context)?;
                this.write_scalar(result, dest)?;
            }
            "SleepConditionVariableSRW" => {
                let [condvar, lock, timeout, flags] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;

                let result = this.SleepConditionVariableSRW(condvar, lock, timeout, flags, dest)?;
                this.write_scalar(result, dest)?;
            }
            "WakeConditionVariable" => {
                let [condvar] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;

                this.WakeConditionVariable(condvar)?;
            }
            "WakeAllConditionVariable" => {
                let [condvar] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;

                this.WakeAllConditionVariable(condvar)?;
            }

            // Dynamic symbol loading
            "GetProcAddress" => {
                #[allow(non_snake_case)]
                let [hModule, lpProcName] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.read_target_isize(hModule)?;
                let name = this.read_c_str(this.read_pointer(lpProcName)?)?;
                if let Some(dlsym) = Dlsym::from_str(name, &this.tcx.sess.target.os)? {
                    let ptr = this.create_fn_alloc_ptr(FnVal::Other(dlsym));
                    this.write_pointer(ptr, dest)?;
                } else {
                    this.write_null(dest)?;
                }
            }

            // Miscellaneous
            "SystemFunction036" => {
                // This is really 'RtlGenRandom'.
                let [ptr, len] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                let len = this.read_scalar(len)?.to_u32()?;
                this.gen_random(ptr, len.into())?;
                this.write_scalar(Scalar::from_bool(true), dest)?;
            }
            "BCryptGenRandom" => {
                let [algorithm, ptr, len, flags] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let algorithm = this.read_scalar(algorithm)?;
                let algorithm = algorithm.to_target_usize(this)?;
                let ptr = this.read_pointer(ptr)?;
                let len = this.read_scalar(len)?.to_u32()?;
                let flags = this.read_scalar(flags)?.to_u32()?;
                match flags {
                    0 => {
                        if algorithm != 0x81 {
                            // BCRYPT_RNG_ALG_HANDLE
                            throw_unsup_format!(
                                "BCryptGenRandom algorithm must be BCRYPT_RNG_ALG_HANDLE when the flag is 0"
                            );
                        }
                    }
                    2 => {
                        // BCRYPT_USE_SYSTEM_PREFERRED_RNG
                        if algorithm != 0 {
                            throw_unsup_format!(
                                "BCryptGenRandom algorithm must be NULL when the flag is BCRYPT_USE_SYSTEM_PREFERRED_RNG"
                            );
                        }
                    }
                    _ => {
                        throw_unsup_format!(
                            "BCryptGenRandom is only supported with BCRYPT_USE_SYSTEM_PREFERRED_RNG or BCRYPT_RNG_ALG_HANDLE"
                        );
                    }
                }
                this.gen_random(ptr, len.into())?;
                this.write_null(dest)?; // STATUS_SUCCESS
            }
            "GetConsoleScreenBufferInfo" => {
                // `term` needs this, so we fake it.
                let [console, buffer_info] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.read_target_isize(console)?;
                // FIXME: this should use deref_operand_as, but CONSOLE_SCREEN_BUFFER_INFO is not in std
                this.deref_operand(buffer_info)?;
                // Indicate an error.
                // FIXME: we should set last_error, but to what?
                this.write_null(dest)?;
            }
            "GetStdHandle" => {
                let [which] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let which = this.read_scalar(which)?.to_i32()?;
                // We just make this the identity function, so we know later in `NtWriteFile` which
                // one it is. This is very fake, but libtest needs it so we cannot make it a
                // std-only shim.
                // FIXME: this should return real HANDLEs when io support is added
                this.write_scalar(Scalar::from_target_isize(which.into(), this), dest)?;
            }
            "CloseHandle" => {
                let [handle] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;

                this.CloseHandle(handle)?;

                this.write_scalar(Scalar::from_u32(1), dest)?;
            }
            "GetModuleFileNameW" => {
                let [handle, filename, size] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.check_no_isolation("`GetModuleFileNameW`")?;

                let handle = this.read_target_usize(handle)?;
                let filename = this.read_pointer(filename)?;
                let size = this.read_scalar(size)?.to_u32()?;

                if handle != 0 {
                    throw_unsup_format!("`GetModuleFileNameW` only supports the NULL handle");
                }

                // Using the host current_exe is a bit off, but consistent with Linux
                // (where stdlib reads /proc/self/exe).
                // Unfortunately this Windows function has a crazy behavior so we can't just use
                // `write_path_to_wide_str`...
                let path = std::env::current_exe().unwrap();
                let (all_written, size_needed) = this.write_path_to_wide_str(
                    &path,
                    filename,
                    size.into(),
                    /*truncate*/ true,
                )?;

                if all_written {
                    // If the function succeeds, the return value is the length of the string that
                    // is copied to the buffer, in characters, not including the terminating null
                    // character.
                    this.write_int(size_needed.checked_sub(1).unwrap(), dest)?;
                } else {
                    // If the buffer is too small to hold the module name, the string is truncated
                    // to nSize characters including the terminating null character, the function
                    // returns nSize, and the function sets the last error to
                    // ERROR_INSUFFICIENT_BUFFER.
                    this.write_int(size, dest)?;
                    let insufficient_buffer = this.eval_windows("c", "ERROR_INSUFFICIENT_BUFFER");
                    this.set_last_error(insufficient_buffer)?;
                }
            }

            // Threading
            "CreateThread" => {
                let [security, stacksize, start, arg, flags, thread] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;

                let thread_id =
                    this.CreateThread(security, stacksize, start, arg, flags, thread)?;

                this.write_scalar(Handle::Thread(thread_id).to_scalar(this), dest)?;
            }
            "WaitForSingleObject" => {
                let [handle, timeout] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;

                let ret = this.WaitForSingleObject(handle, timeout)?;
                this.write_scalar(Scalar::from_u32(ret), dest)?;
            }
            "GetCurrentThread" => {
                let [] = this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;

                this.write_scalar(
                    Handle::Pseudo(PseudoHandle::CurrentThread).to_scalar(this),
                    dest,
                )?;
            }

            // Incomplete shims that we "stub out" just to get pre-main initialization code to work.
            // These shims are enabled only when the caller is in the standard library.
            "GetProcessHeap" if this.frame_in_std() => {
                let [] = this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                // Just fake a HANDLE
                // It's fine to not use the Handle type here because its a stub
                this.write_int(1, dest)?;
            }
            "GetModuleHandleA" if this.frame_in_std() => {
                #[allow(non_snake_case)]
                let [_lpModuleName] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                // We need to return something non-null here to make `compat_fn!` work.
                this.write_int(1, dest)?;
            }
            "SetConsoleTextAttribute" if this.frame_in_std() => {
                #[allow(non_snake_case)]
                let [_hConsoleOutput, _wAttribute] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                // Pretend these does not exist / nothing happened, by returning zero.
                this.write_null(dest)?;
            }
            "GetConsoleMode" if this.frame_in_std() => {
                let [console, mode] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.read_target_isize(console)?;
                this.deref_operand(mode)?;
                // Indicate an error.
                this.write_null(dest)?;
            }
            "GetFileType" if this.frame_in_std() => {
                #[allow(non_snake_case)]
                let [_hFile] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                // Return unknown file type.
                this.write_null(dest)?;
            }
            "AddVectoredExceptionHandler" if this.frame_in_std() => {
                #[allow(non_snake_case)]
                let [_First, _Handler] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                // Any non zero value works for the stdlib. This is just used for stack overflows anyway.
                this.write_int(1, dest)?;
            }
            "SetThreadStackGuarantee" if this.frame_in_std() => {
                #[allow(non_snake_case)]
                let [_StackSizeInBytes] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                // Any non zero value works for the stdlib. This is just used for stack overflows anyway.
                this.write_int(1, dest)?;
            }
            "GetCurrentProcessId" if this.frame_in_std() => {
                let [] = this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let result = this.GetCurrentProcessId()?;
                this.write_int(result, dest)?;
            }
            // this is only callable from std because we know that std ignores the return value
            "SwitchToThread" if this.frame_in_std() => {
                let [] = this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;

                this.yield_active_thread();

                // FIXME: this should return a nonzero value if this call does result in switching to another thread.
                this.write_null(dest)?;
            }

            _ => return Ok(EmulateByNameResult::NotSupported),
        }

        Ok(EmulateByNameResult::NeedsJumping)
    }
}
