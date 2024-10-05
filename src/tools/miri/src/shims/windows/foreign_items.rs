use std::ffi::OsStr;
use std::path::{self, Path, PathBuf};
use std::{io, iter, str};

use rustc_span::Symbol;
use rustc_target::abi::{Align, Size};
use rustc_target::spec::abi::Abi;

use self::shims::windows::handle::{Handle, PseudoHandle};
use crate::shims::os_str::bytes_to_os_str;
use crate::shims::windows::*;
use crate::*;

pub fn is_dyn_sym(name: &str) -> bool {
    // std does dynamic detection for these symbols
    matches!(
        name,
        "SetThreadDescription" | "GetThreadDescription" | "WaitOnAddress" | "WakeByAddressSingle"
    )
}

#[cfg(windows)]
fn win_absolute<'tcx>(path: &Path) -> InterpResult<'tcx, io::Result<PathBuf>> {
    // We are on Windows so we can simply let the host do this.
    interp_ok(path::absolute(path))
}

#[cfg(unix)]
#[allow(clippy::get_first, clippy::arithmetic_side_effects)]
fn win_absolute<'tcx>(path: &Path) -> InterpResult<'tcx, io::Result<PathBuf>> {
    // We are on Unix, so we need to implement parts of the logic ourselves.
    let bytes = path.as_os_str().as_encoded_bytes();
    // If it starts with `//` (these were backslashes but are already converted)
    // then this is a magic special path, we just leave it unchanged.
    if bytes.get(0).copied() == Some(b'/') && bytes.get(1).copied() == Some(b'/') {
        return interp_ok(Ok(path.into()));
    };
    // Special treatment for Windows' magic filenames: they are treated as being relative to `\\.\`.
    let magic_filenames = &[
        "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
        "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    ];
    if magic_filenames.iter().any(|m| m.as_bytes() == bytes) {
        let mut result: Vec<u8> = br"//./".into();
        result.extend(bytes);
        return interp_ok(Ok(bytes_to_os_str(&result)?.into()));
    }
    // Otherwise we try to do something kind of close to what Windows does, but this is probably not
    // right in all cases. We iterate over the components between `/`, and remove trailing `.`,
    // except that trailing `..` remain unchanged.
    let mut result = vec![];
    let mut bytes = bytes; // the remaining bytes to process
    loop {
        let len = bytes.iter().position(|&b| b == b'/').unwrap_or(bytes.len());
        let mut component = &bytes[..len];
        if len >= 2 && component[len - 1] == b'.' && component[len - 2] != b'.' {
            // Strip trailing `.`
            component = &component[..len - 1];
        }
        // Add this component to output.
        result.extend(component);
        // Prepare next iteration.
        if len < bytes.len() {
            // There's a component after this; add `/` and process remaining bytes.
            result.push(b'/');
            bytes = &bytes[len + 1..];
            continue;
        } else {
            // This was the last component and it did not have a trailing `/`.
            break;
        }
    }
    // Let the host `absolute` function do working-dir handling
    interp_ok(path::absolute(bytes_to_os_str(&result)?))
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn emulate_foreign_item_inner(
        &mut self,
        link_name: Symbol,
        abi: Abi,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();

        // See `fn emulate_foreign_item_inner` in `shims/foreign_items.rs` for the general pattern.

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
            "GetUserProfileDirectoryW" => {
                let [token, buf, size] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let result = this.GetUserProfileDirectoryW(token, buf, size)?;
                this.write_scalar(result, dest)?;
            }
            "GetCurrentProcessId" => {
                let [] = this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let result = this.GetCurrentProcessId()?;
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
                    .deref_pointer_as(io_status_block, this.windows_ty_layout("IO_STATUS_BLOCK"))?;

                if byte_offset != 0 {
                    throw_unsup_format!(
                        "`NtWriteFile` `ByteOffset` parameter is non-null, which is unsupported"
                    );
                }

                let written = if handle == -11 || handle == -12 {
                    // stdout/stderr
                    use io::Write;

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
                        &io_status_information,
                    )?;
                }
                // Return whether this was a success. >= 0 is success.
                // For the error code we arbitrarily pick 0xC0000185, STATUS_IO_DEVICE_ERROR.
                this.write_scalar(
                    Scalar::from_u32(if written.is_some() { 0 } else { 0xC0000185u32 }),
                    dest,
                )?;
            }
            "GetFullPathNameW" => {
                let [filename, size, buffer, filepart] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.check_no_isolation("`GetFullPathNameW`")?;

                let filename = this.read_pointer(filename)?;
                let size = this.read_scalar(size)?.to_u32()?;
                let buffer = this.read_pointer(buffer)?;
                let filepart = this.read_pointer(filepart)?;

                if !this.ptr_is_null(filepart)? {
                    throw_unsup_format!("GetFullPathNameW: non-null `lpFilePart` is not supported");
                }

                let filename = this.read_path_from_wide_str(filename)?;
                let result = match win_absolute(&filename)? {
                    Err(err) => {
                        this.set_last_error(err)?;
                        Scalar::from_u32(0) // return zero upon failure
                    }
                    Ok(abs_filename) => {
                        Scalar::from_u32(helpers::windows_check_buffer_size(
                            this.write_path_to_wide_str(&abs_filename, buffer, size.into())?,
                        ))
                        // This can in fact return 0. It is up to the caller to set last_error to 0
                        // beforehand and check it afterwards to exclude that case.
                    }
                };
                this.write_scalar(result, dest)?;
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
                // Alignment is twice the pointer size.
                // Source: <https://learn.microsoft.com/en-us/windows/win32/api/heapapi/nf-heapapi-heapalloc>
                let align = this.tcx.pointer_size().bytes().strict_mul(2);
                let ptr = this.allocate_ptr(
                    Size::from_bytes(size),
                    Align::from_bytes(align).unwrap(),
                    MiriMemoryKind::WinHeap.into(),
                )?;
                if zero_init {
                    this.write_bytes_ptr(
                        ptr.into(),
                        iter::repeat(0u8).take(usize::try_from(size).unwrap()),
                    )?;
                }
                this.write_pointer(ptr, dest)?;
            }
            "HeapFree" => {
                let [handle, flags, ptr] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.read_target_isize(handle)?;
                this.read_scalar(flags)?.to_u32()?;
                let ptr = this.read_pointer(ptr)?;
                // "This pointer can be NULL." It doesn't say what happens then, but presumably nothing.
                // (https://learn.microsoft.com/en-us/windows/win32/api/heapapi/nf-heapapi-heapfree)
                if !this.ptr_is_null(ptr)? {
                    this.deallocate_ptr(ptr, None, MiriMemoryKind::WinHeap.into())?;
                }
                this.write_scalar(Scalar::from_i32(1), dest)?;
            }
            "HeapReAlloc" => {
                let [handle, flags, old_ptr, size] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.read_target_isize(handle)?;
                this.read_scalar(flags)?.to_u32()?;
                let old_ptr = this.read_pointer(old_ptr)?;
                let size = this.read_target_usize(size)?;
                let align = this.tcx.pointer_size().bytes().strict_mul(2); // same as above
                // The docs say that `old_ptr` must come from an earlier HeapAlloc or HeapReAlloc,
                // so unlike C `realloc` we do *not* allow a NULL here.
                // (https://learn.microsoft.com/en-us/windows/win32/api/heapapi/nf-heapapi-heaprealloc)
                let new_ptr = this.reallocate_ptr(
                    old_ptr,
                    None,
                    Size::from_bytes(size),
                    Align::from_bytes(align).unwrap(),
                    MiriMemoryKind::WinHeap.into(),
                )?;
                this.write_pointer(new_ptr, dest)?;
            }
            "LocalFree" => {
                let [ptr] = this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                // "If the hMem parameter is NULL, LocalFree ignores the parameter and returns NULL."
                // (https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-localfree)
                if !this.ptr_is_null(ptr)? {
                    this.deallocate_ptr(ptr, None, MiriMemoryKind::WinLocal.into())?;
                }
                this.write_null(dest)?;
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
                    this.deref_pointer_as(system_info, this.windows_ty_layout("SYSTEM_INFO"))?;
                // Initialize with `0`.
                this.write_bytes_ptr(
                    system_info.ptr(),
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
                let active_thread = this.active_thread();
                let ptr = this.machine.tls.load_tls(key, active_thread, this)?;
                this.write_scalar(ptr, dest)?;
            }
            "TlsSetValue" => {
                let [key, new_ptr] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let key = u128::from(this.read_scalar(key)?.to_u32()?);
                let active_thread = this.active_thread();
                let new_data = this.read_scalar(new_ptr)?;
                this.machine.tls.store_tls(key, active_thread, new_data, &*this.tcx)?;

                // Return success (`1`).
                this.write_int(1, dest)?;
            }

            // Access to command-line arguments
            "GetCommandLineW" => {
                let [] = this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.write_pointer(
                    this.machine.cmd_line.expect("machine must be initialized"),
                    dest,
                )?;
            }

            // Time related shims
            "GetSystemTimeAsFileTime" | "GetSystemTimePreciseAsFileTime" => {
                #[allow(non_snake_case)]
                let [LPFILETIME] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.GetSystemTimeAsFileTime(link_name.as_str(), LPFILETIME)?;
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
            "CreateWaitableTimerExW" => {
                let [attributes, name, flags, access] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.read_pointer(attributes)?;
                this.read_pointer(name)?;
                this.read_scalar(flags)?.to_u32()?;
                this.read_scalar(access)?.to_u32()?;
                // Unimplemented. Always return failure.
                let not_supported = this.eval_windows("c", "ERROR_NOT_SUPPORTED");
                this.set_last_error(not_supported)?;
                this.write_null(dest)?;
            }

            // Synchronization primitives
            "InitOnceBeginInitialize" => {
                let [ptr, flags, pending, context] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.InitOnceBeginInitialize(ptr, flags, pending, context, dest)?;
            }
            "InitOnceComplete" => {
                let [ptr, flags, context] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let result = this.InitOnceComplete(ptr, flags, context)?;
                this.write_scalar(result, dest)?;
            }
            "WaitOnAddress" => {
                let [ptr_op, compare_op, size_op, timeout_op] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;

                this.WaitOnAddress(ptr_op, compare_op, size_op, timeout_op, dest)?;
            }
            "WakeByAddressSingle" => {
                let [ptr_op] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;

                this.WakeByAddressSingle(ptr_op)?;
            }
            "WakeByAddressAll" => {
                let [ptr_op] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;

                this.WakeByAddressAll(ptr_op)?;
            }

            // Dynamic symbol loading
            "GetProcAddress" => {
                #[allow(non_snake_case)]
                let [hModule, lpProcName] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                this.read_target_isize(hModule)?;
                let name = this.read_c_str(this.read_pointer(lpProcName)?)?;
                if let Ok(name) = str::from_utf8(name)
                    && is_dyn_sym(name)
                {
                    let ptr = this.fn_ptr(FnVal::Other(DynSym::from_str(name)));
                    this.write_pointer(ptr, dest)?;
                } else {
                    this.write_null(dest)?;
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
            "SetThreadDescription" => {
                let [handle, name] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;

                let handle = this.read_scalar(handle)?;
                let name = this.read_wide_str(this.read_pointer(name)?)?;

                let thread = match Handle::from_scalar(handle, this)? {
                    Some(Handle::Thread(thread)) => thread,
                    Some(Handle::Pseudo(PseudoHandle::CurrentThread)) => this.active_thread(),
                    _ => this.invalid_handle("SetThreadDescription")?,
                };

                // FIXME: use non-lossy conversion
                this.set_thread_name(thread, String::from_utf16_lossy(&name).into_bytes());

                this.write_null(dest)?;
            }
            "GetThreadDescription" => {
                let [handle, name_ptr] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;

                let handle = this.read_scalar(handle)?;
                let name_ptr = this.deref_pointer(name_ptr)?; // the pointer where we should store the ptr to the name

                let thread = match Handle::from_scalar(handle, this)? {
                    Some(Handle::Thread(thread)) => thread,
                    Some(Handle::Pseudo(PseudoHandle::CurrentThread)) => this.active_thread(),
                    _ => this.invalid_handle("SetThreadDescription")?,
                };
                // Looks like the default thread name is empty.
                let name = this.get_thread_name(thread).unwrap_or(b"").to_owned();
                let name = this.alloc_os_str_as_wide_str(
                    bytes_to_os_str(&name)?,
                    MiriMemoryKind::WinLocal.into(),
                )?;

                this.write_scalar(Scalar::from_maybe_pointer(name, this), &name_ptr)?;

                this.write_null(dest)?;
            }

            // Miscellaneous
            "ExitProcess" => {
                let [code] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let code = this.read_scalar(code)?.to_u32()?;
                throw_machine_stop!(TerminationInfo::Exit { code: code.into(), leak_check: false });
            }
            "SystemFunction036" => {
                // used by getrandom 0.1
                // This is really 'RtlGenRandom'.
                let [ptr, len] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                let len = this.read_scalar(len)?.to_u32()?;
                this.gen_random(ptr, len.into())?;
                this.write_scalar(Scalar::from_bool(true), dest)?;
            }
            "ProcessPrng" => {
                // used by `std`
                let [ptr, len] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                let len = this.read_target_usize(len)?;
                this.gen_random(ptr, len)?;
                this.write_int(1, dest)?;
            }
            "BCryptGenRandom" => {
                // used by getrandom 0.2
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
                // FIXME: this should use deref_pointer_as, but CONSOLE_SCREEN_BUFFER_INFO is not in std
                this.deref_pointer(buffer_info)?;
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

                this.write_int(1, dest)?;
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
                let path = std::env::current_exe().unwrap();
                let (all_written, size_needed) =
                    this.write_path_to_wide_str_truncated(&path, filename, size.into())?;

                if all_written {
                    // If the function succeeds, the return value is the length of the string that
                    // is copied to the buffer, in characters, not including the terminating null
                    // character.
                    this.write_int(size_needed.strict_sub(1), dest)?;
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
            "FormatMessageW" => {
                let [flags, module, message_id, language_id, buffer, size, arguments] =
                    this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;

                let flags = this.read_scalar(flags)?.to_u32()?;
                let _module = this.read_pointer(module)?; // seems to contain a module name
                let message_id = this.read_scalar(message_id)?;
                let _language_id = this.read_scalar(language_id)?.to_u32()?;
                let buffer = this.read_pointer(buffer)?;
                let size = this.read_scalar(size)?.to_u32()?;
                let _arguments = this.read_pointer(arguments)?;

                // We only support `FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS`
                // This also means `arguments` can be ignored.
                if flags != 4096u32 | 512u32 {
                    throw_unsup_format!("FormatMessageW: unsupported flags {flags:#x}");
                }

                let error = this.try_errnum_to_io_error(message_id)?;
                let formatted = match error {
                    Some(err) => format!("{err}"),
                    None => format!("<unknown error in FormatMessageW: {message_id}>"),
                };
                let (complete, length) =
                    this.write_os_str_to_wide_str(OsStr::new(&formatted), buffer, size.into())?;
                if !complete {
                    // The API docs don't say what happens when the buffer is not big enough...
                    // Let's just bail.
                    throw_unsup_format!("FormatMessageW: buffer not big enough");
                }
                // The return value is the number of characters stored *excluding* the null terminator.
                this.write_int(length.strict_sub(1), dest)?;
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
                this.deref_pointer(mode)?;
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
            // this is only callable from std because we know that std ignores the return value
            "SwitchToThread" if this.frame_in_std() => {
                let [] = this.check_shim(abi, Abi::System { unwind: false }, link_name, args)?;

                this.yield_active_thread();

                // FIXME: this should return a nonzero value if this call does result in switching to another thread.
                this.write_null(dest)?;
            }

            "_Unwind_RaiseException" => {
                // This is not formally part of POSIX, but it is very wide-spread on POSIX systems.
                // It was originally specified as part of the Itanium C++ ABI:
                // https://itanium-cxx-abi.github.io/cxx-abi/abi-eh.html#base-throw.
                // MinGW implements _Unwind_RaiseException on top of SEH exceptions.
                if this.tcx.sess.target.env != "gnu" {
                    throw_unsup_format!(
                        "`_Unwind_RaiseException` is not supported on non-MinGW Windows",
                    );
                }
                // This function looks and behaves excatly like miri_start_unwind.
                let [payload] = this.check_shim(abi, Abi::C { unwind: true }, link_name, args)?;
                this.handle_miri_start_unwind(payload)?;
                return interp_ok(EmulateItemResult::NeedsUnwind);
            }

            _ => return interp_ok(EmulateItemResult::NotSupported),
        }

        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
