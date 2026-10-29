use std::ffi::OsStr;
use std::path::{self, Path, PathBuf};
use std::{io, iter, str};

use rustc_abi::{Align, Size};
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;
use rustc_target::spec::Env;

use self::shims::windows::handle::{Handle, PseudoHandle};
use crate::eval::{ExitType, ProgramExitHandler};
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
fn win_get_full_path_name<'tcx>(path: &Path) -> InterpResult<'tcx, io::Result<PathBuf>> {
    // We are on Windows so we can simply let the host do this.
    interp_ok(path::absolute(path))
}

#[cfg(unix)]
#[expect(clippy::get_first, clippy::arithmetic_side_effects)]
fn win_get_full_path_name<'tcx>(path: &Path) -> InterpResult<'tcx, io::Result<PathBuf>> {
    use std::sync::LazyLock;

    use rustc_data_structures::fx::FxHashSet;

    // We are on Unix, so we need to implement parts of the logic ourselves. `path` will use `/`
    // separators, and the result should also use `/`.
    // See <https://chrisdenton.github.io/omnipath/Overview.html#absolute-win32-paths> for more
    // information about Windows paths.
    // This does not handle all corner cases correctly, see
    // <https://github.com/rust-lang/miri/pull/4262#issuecomment-2792168853> for more cursed
    // examples.
    let bytes = path.as_os_str().as_encoded_bytes();
    // If it starts with `//./` or `//?/` then this is a magic special path, we just leave it
    // unchanged.
    if bytes.get(0).copied() == Some(b'/')
        && bytes.get(1).copied() == Some(b'/')
        && matches!(bytes.get(2), Some(b'.' | b'?'))
        && bytes.get(3).copied() == Some(b'/')
    {
        return interp_ok(Ok(path.into()));
    };
    let is_unc = bytes.starts_with(b"//");
    // Special treatment for Windows' magic filenames: they are treated as being relative to `//./`.
    static MAGIC_FILENAMES: LazyLock<FxHashSet<&'static str>> = LazyLock::new(|| {
        FxHashSet::from_iter([
            "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7",
            "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
        ])
    });
    if str::from_utf8(bytes).is_ok_and(|s| MAGIC_FILENAMES.contains(&*s.to_ascii_uppercase())) {
        let mut result: Vec<u8> = b"//./".into();
        result.extend(bytes);
        return interp_ok(Ok(bytes_to_os_str(&result)?.into()));
    }
    // Otherwise we try to do something kind of close to what Windows does, but this is probably not
    // right in all cases.
    let mut result: Vec<&[u8]> = vec![]; // will be a vector of components, joined by `/`.
    let mut bytes = bytes; // the remaining bytes to process
    let mut stop = false;
    while !stop {
        // Find next component, and advance `bytes`.
        let mut component = match bytes.iter().position(|&b| b == b'/') {
            Some(pos) => {
                let (component, tail) = bytes.split_at(pos);
                bytes = &tail[1..]; // remove the `/`.
                component
            }
            None => {
                // There's no more `/`.
                stop = true;
                let component = bytes;
                bytes = &[];
                component
            }
        };
        // `NUL` and only `NUL` also gets changed to be relative to `//./` later in the path.
        // (This changed with Windows 11; previously, all magic filenames behaved like this.)
        // Also, this does not apply to UNC paths.
        if !is_unc && component.eq_ignore_ascii_case(b"NUL") {
            let mut result: Vec<u8> = b"//./".into();
            result.extend(component);
            return interp_ok(Ok(bytes_to_os_str(&result)?.into()));
        }
        // Deal with `..` -- Windows handles this entirely syntactically.
        if component == b".." {
            // Remove previous component, unless we are at the "root" already, then just ignore the `..`.
            let is_root = {
                // Paths like `/C:`.
                result.len() == 2 && matches!(result[0], []) && matches!(result[1], [_, b':'])
            } || {
                // Paths like `//server/share`
                result.len() == 4 && matches!(result[0], []) && matches!(result[1], [])
            };
            if !is_root {
                result.pop();
            }
            continue;
        }
        // Preserve this component.
        // Strip trailing `.`, but preserve trailing `..`. But not for UNC paths!
        let len = component.len();
        if !is_unc && len >= 2 && component[len - 1] == b'.' && component[len - 2] != b'.' {
            component = &component[..len - 1];
        }
        // Add this component to output.
        result.push(component);
    }
    // Drive letters must be followed by a `/`.
    if result.len() == 2 && matches!(result[0], []) && matches!(result[1], [_, b':']) {
        result.push(&[]);
    }
    // Let the host `absolute` function do working-dir handling.
    let result = result.join(&b'/');
    interp_ok(path::absolute(bytes_to_os_str(&result)?))
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn emulate_foreign_item_inner(
        &mut self,
        link_name: Symbol,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let ecx = self.eval_context_mut();

        // See `fn emulate_foreign_item_inner` in `shims/foreign_items.rs` for the general pattern.

        // Windows API stubs.
        // HANDLE = *mut c_void (formerly: isize)
        // NTSTATUS = LONG = i32
        // DWORD = ULONG = u32
        // BOOL = i32
        // BOOLEAN = u8
        match link_name.as_str() {
            // Environment related shims
            "GetEnvironmentVariableW" => {
                // FIXME: This does not have a direct test (#3179).
                let [name, buf, size] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(*_, *_, u32) -> u32),
                    (link_name, abi, args),
                )?;
                let result = ecx.GetEnvironmentVariableW(name, buf, size)?;
                ecx.write_scalar(result, dest)?;
            }
            "SetEnvironmentVariableW" => {
                // FIXME: This does not have a direct test (#3179).
                let [name, value] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(*_, *_) -> winapi::BOOL),
                    (link_name, abi, args),
                )?;
                let result = ecx.SetEnvironmentVariableW(name, value)?;
                ecx.write_scalar(result, dest)?;
            }
            "GetEnvironmentStringsW" => {
                // FIXME: This does not have a direct test (#3179).
                let [] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn() -> *_),
                    (link_name, abi, args),
                )?;
                let result = ecx.GetEnvironmentStringsW()?;
                ecx.write_pointer(result, dest)?;
            }
            "FreeEnvironmentStringsW" => {
                // FIXME: This does not have a direct test (#3179).
                let [env_block] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(*_) -> winapi::BOOL),
                    (link_name, abi, args),
                )?;
                let result = ecx.FreeEnvironmentStringsW(env_block)?;
                ecx.write_scalar(result, dest)?;
            }
            "GetCurrentDirectoryW" => {
                // FIXME: This does not have a direct test (#3179).
                let [size, buf] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(u32, *_) -> u32),
                    (link_name, abi, args),
                )?;
                let result = ecx.GetCurrentDirectoryW(size, buf)?;
                ecx.write_scalar(result, dest)?;
            }
            "SetCurrentDirectoryW" => {
                // FIXME: This does not have a direct test (#3179).
                let [path] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(*_) -> winapi::BOOL),
                    (link_name, abi, args),
                )?;
                let result = ecx.SetCurrentDirectoryW(path)?;
                ecx.write_scalar(result, dest)?;
            }
            "GetUserProfileDirectoryW" => {
                // FIXME: This does not have a direct test (#3179).
                let [token, buf, size] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(winapi::HANDLE, *_, *_) -> winapi::BOOL),
                    (link_name, abi, args),
                )?;
                let result = ecx.GetUserProfileDirectoryW(token, buf, size)?;
                ecx.write_scalar(result, dest)?;
            }
            "GetCurrentProcessId" => {
                // FIXME: This does not have a direct test (#3179).
                let [] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn() -> u32),
                    (link_name, abi, args),
                )?;
                let result = ecx.GetCurrentProcessId()?;
                ecx.write_scalar(result, dest)?;
            }
            "GetTempPathW" => {
                // FIXME: This does not have a direct test (#3179).
                let [bufferlength, buffer] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(u32, *_) -> u32),
                    (link_name, abi, args),
                )?;
                let result = ecx.GetTempPathW(bufferlength, buffer)?;
                ecx.write_scalar(result, dest)?;
            }

            // File related shims
            "NtWriteFile" => {
                let [
                    handle,
                    event,
                    apc_routine,
                    apc_context,
                    io_status_block,
                    buf,
                    n,
                    byte_offset,
                    key,
                ] = &ecx.check_shim_sig(
                    shim_sig!(
                        extern "system" fn(
                            winapi::HANDLE,
                            winapi::HANDLE,
                            *_,
                            *_,
                            *_,
                            *_,
                            u32,
                            *_,
                            *_,
                        ) -> i32
                    ),
                    (link_name, abi, args),
                )?;
                ecx.NtWriteFile(
                    handle,
                    event,
                    apc_routine,
                    apc_context,
                    io_status_block,
                    buf,
                    n,
                    byte_offset,
                    key,
                    dest,
                )?;
            }
            "NtReadFile" => {
                let [
                    handle,
                    event,
                    apc_routine,
                    apc_context,
                    io_status_block,
                    buf,
                    n,
                    byte_offset,
                    key,
                ] = &ecx.check_shim_sig(
                    shim_sig!(
                        extern "system" fn(
                            winapi::HANDLE,
                            winapi::HANDLE,
                            *_,
                            *_,
                            *_,
                            *_,
                            u32,
                            *_,
                            *_,
                        ) -> i32
                    ),
                    (link_name, abi, args),
                )?;
                ecx.NtReadFile(
                    handle,
                    event,
                    apc_routine,
                    apc_context,
                    io_status_block,
                    buf,
                    n,
                    byte_offset,
                    key,
                    dest,
                )?;
            }
            "GetFullPathNameW" => {
                // FIXME: This does not have a direct test (#3179).
                let [filename, size, buffer, filepart] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(*_, u32, *_, *_) -> u32),
                    (link_name, abi, args),
                )?;
                ecx.check_no_isolation("`GetFullPathNameW`")?;

                let filename = ecx.read_pointer(filename)?;
                let size = ecx.read_scalar(size)?.to_u32()?;
                let buffer = ecx.read_pointer(buffer)?;
                let filepart = ecx.read_pointer(filepart)?;

                if !ecx.ptr_is_null(filepart)? {
                    throw_unsup_format!("GetFullPathNameW: non-null `lpFilePart` is not supported");
                }

                let filename = ecx.read_path_from_wide_str(filename)?;
                let result = match win_get_full_path_name(&filename)? {
                    Err(err) => {
                        ecx.set_last_error(err)?;
                        Scalar::from_u32(0) // return zero upon failure
                    }
                    Ok(abs_filename) => {
                        Scalar::from_u32(helpers::windows_check_buffer_size(
                            ecx.write_path_to_wide_str(&abs_filename, buffer, size.into())?,
                        ))
                        // This can in fact return 0. It is up to the caller to set last_error to 0
                        // beforehand and check it afterwards to exclude that case.
                    }
                };
                ecx.write_scalar(result, dest)?;
            }
            "CreateFileW" => {
                let [
                    file_name,
                    desired_access,
                    share_mode,
                    security_attributes,
                    creation_disposition,
                    flags_and_attributes,
                    template_file,
                ] = &ecx.check_shim_sig(
                    shim_sig!(
                        extern "system" fn(
                            *_,
                            u32,
                            u32,
                            *_,
                            u32,
                            u32,
                            winapi::HANDLE,
                        ) -> winapi::HANDLE
                    ),
                    (link_name, abi, args),
                )?;
                let handle = ecx.CreateFileW(
                    file_name,
                    desired_access,
                    share_mode,
                    security_attributes,
                    creation_disposition,
                    flags_and_attributes,
                    template_file,
                )?;
                ecx.write_scalar(handle.to_scalar(ecx), dest)?;
            }
            "GetFileInformationByHandle" => {
                let [handle, info] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(winapi::HANDLE, *_) -> winapi::BOOL),
                    (link_name, abi, args),
                )?;
                let res = ecx.GetFileInformationByHandle(handle, info)?;
                ecx.write_scalar(res, dest)?;
            }
            "SetFileInformationByHandle" => {
                let [handle, class, info, size] = &ecx.check_shim_sig(
                    shim_sig!(
                        extern "system" fn(
                            winapi::HANDLE,
                            winapi::FILE_INFO_BY_HANDLE_CLASS,
                            *_,
                            u32,
                        ) -> winapi::BOOL
                    ),
                    (link_name, abi, args),
                )?;
                let res = ecx.SetFileInformationByHandle(handle, class, info, size)?;
                ecx.write_scalar(res, dest)?;
            }
            "FlushFileBuffers" => {
                let [handle] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(winapi::HANDLE) -> winapi::BOOL),
                    (link_name, abi, args),
                )?;
                let res = ecx.FlushFileBuffers(handle)?;
                ecx.write_scalar(res, dest)?;
            }
            "DeleteFileW" => {
                let [file_name] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(*_) -> winapi::BOOL),
                    (link_name, abi, args),
                )?;
                let res = ecx.DeleteFileW(file_name)?;
                ecx.write_scalar(res, dest)?;
            }
            "SetFilePointerEx" => {
                let [file, distance_to_move, new_file_pointer, move_method] = &ecx.check_shim_sig(
                    // i64 is actually a LARGE_INTEGER union of {u32, i32} and {i64}
                    shim_sig!(extern "system" fn(winapi::HANDLE, i64, *_, u32) -> winapi::BOOL),
                    (link_name, abi, args),
                )?;
                let res =
                    ecx.SetFilePointerEx(file, distance_to_move, new_file_pointer, move_method)?;
                ecx.write_scalar(res, dest)?;
            }
            "MoveFileExW" => {
                let [existing_name, new_name, flags] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(*_, *_, u32) -> winapi::BOOL),
                    (link_name, abi, args),
                )?;
                let res = ecx.MoveFileExW(existing_name, new_name, flags)?;
                ecx.write_scalar(res, dest)?;
            }

            // Allocation
            "HeapAlloc" => {
                // FIXME: This does not have a direct test (#3179).
                let [handle, flags, size] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(winapi::HANDLE, u32, usize) -> *_),
                    (link_name, abi, args),
                )?;
                ecx.read_target_isize(handle)?;
                let flags = ecx.read_scalar(flags)?.to_u32()?;
                let size = ecx.read_target_usize(size)?;
                const HEAP_ZERO_MEMORY: u32 = 0x00000008;
                let init = if (flags & HEAP_ZERO_MEMORY) == HEAP_ZERO_MEMORY {
                    AllocInit::Zero
                } else {
                    AllocInit::Uninit
                };
                // Alignment is twice the pointer size.
                // Source: <https://learn.microsoft.com/en-us/windows/win32/api/heapapi/nf-heapapi-heapalloc>
                let align = ecx.tcx.pointer_size().bytes().strict_mul(2);
                let ptr = ecx.allocate_ptr(
                    Size::from_bytes(size),
                    Align::from_bytes(align).unwrap(),
                    MiriMemoryKind::WinHeap.into(),
                    init,
                )?;
                ecx.write_pointer(ptr, dest)?;
            }
            "HeapFree" => {
                // FIXME: This does not have a direct test (#3179).
                let [handle, flags, ptr] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(winapi::HANDLE, u32, *_) -> winapi::BOOL),
                    (link_name, abi, args),
                )?;
                ecx.read_target_isize(handle)?;
                ecx.read_scalar(flags)?.to_u32()?;
                let ptr = ecx.read_pointer(ptr)?;
                // "This pointer can be NULL." It doesn't say what happens then, but presumably nothing.
                // (https://learn.microsoft.com/en-us/windows/win32/api/heapapi/nf-heapapi-heapfree)
                if !ecx.ptr_is_null(ptr)? {
                    ecx.deallocate_ptr(ptr, None, MiriMemoryKind::WinHeap.into())?;
                }
                ecx.write_scalar(Scalar::from_i32(1), dest)?;
            }
            "HeapReAlloc" => {
                // FIXME: This does not have a direct test (#3179).
                let [handle, flags, old_ptr, size] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(winapi::HANDLE, u32, *_, usize) -> *_),
                    (link_name, abi, args),
                )?;
                ecx.read_target_isize(handle)?;
                ecx.read_scalar(flags)?.to_u32()?;
                let old_ptr = ecx.read_pointer(old_ptr)?;
                let size = ecx.read_target_usize(size)?;
                let align = ecx.tcx.pointer_size().bytes().strict_mul(2); // same as above
                // The docs say that `old_ptr` must come from an earlier HeapAlloc or HeapReAlloc,
                // so unlike C `realloc` we do *not* allow a NULL here.
                // (https://learn.microsoft.com/en-us/windows/win32/api/heapapi/nf-heapapi-heaprealloc)
                let new_ptr = ecx.reallocate_ptr(
                    old_ptr,
                    None,
                    Size::from_bytes(size),
                    Align::from_bytes(align).unwrap(),
                    MiriMemoryKind::WinHeap.into(),
                    AllocInit::Uninit,
                )?;
                ecx.write_pointer(new_ptr, dest)?;
            }
            "LocalFree" => {
                let [ptr] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(winapi::HLOCAL) -> winapi::HLOCAL),
                    (link_name, abi, args),
                )?;
                let ptr = ecx.read_pointer(ptr)?;
                // "If the hMem parameter is NULL, LocalFree ignores the parameter and returns NULL."
                // (https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-localfree)
                if !ecx.ptr_is_null(ptr)? {
                    ecx.deallocate_ptr(ptr, None, MiriMemoryKind::WinLocal.into())?;
                }
                ecx.write_null(dest)?;
            }

            // errno
            "SetLastError" => {
                let [error] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(u32) -> ()),
                    (link_name, abi, args),
                )?;
                let error = ecx.read_scalar(error)?;
                ecx.set_last_error(error)?;
            }
            "GetLastError" => {
                let [] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn() -> u32),
                    (link_name, abi, args),
                )?;
                let last_error = ecx.get_last_error()?;
                ecx.write_scalar(last_error, dest)?;
            }
            "RtlNtStatusToDosError" => {
                let [status] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(i32) -> u32),
                    (link_name, abi, args),
                )?;
                let status = ecx.read_scalar(status)?.to_u32()?;
                let err = match status {
                    // STATUS_MEDIA_WRITE_PROTECTED => ERROR_WRITE_PROTECT
                    0xC00000A2 => 19,
                    // STATUS_FILE_INVALID => ERROR_FILE_INVALID
                    0xC0000098 => 1006,
                    // STATUS_DISK_FULL => ERROR_DISK_FULL
                    0xC000007F => 112,
                    // STATUS_IO_DEVICE_ERROR => ERROR_IO_DEVICE
                    0xC0000185 => 1117,
                    // STATUS_ACCESS_DENIED => ERROR_ACCESS_DENIED
                    0xC0000022 => 5,
                    // Anything without an error code => ERROR_MR_MID_NOT_FOUND
                    _ => 317,
                };
                ecx.write_scalar(Scalar::from_i32(err), dest)?;
            }

            // Querying system information
            "GetSystemInfo" => {
                // FIXME: This does not have a direct test (#3179).
                // Also called from `page_size` crate.
                let [system_info] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(*_) -> ()),
                    (link_name, abi, args),
                )?;
                let system_info =
                    ecx.deref_pointer_as(system_info, ecx.windows_ty_layout("SYSTEM_INFO"))?;
                // Initialize with `0`.
                ecx.write_bytes_ptr(
                    system_info.ptr(),
                    iter::repeat_n(0u8, system_info.layout.size.bytes_usize()),
                )?;
                // Set selected fields.
                ecx.write_int_fields_named(
                    &[
                        ("dwPageSize", ecx.machine.page_size.into()),
                        ("dwNumberOfProcessors", ecx.machine.num_cpus.into()),
                    ],
                    &system_info,
                )?;
            }
            // Thread-local storage
            "TlsAlloc" => {
                // This just creates a key; Windows does not natively support TLS destructors.

                // Create key and return it.
                let [] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn() -> u32),
                    (link_name, abi, args),
                )?;
                let key = ecx.machine.tls.create_tls_key(None, dest.layout.size)?;
                ecx.write_scalar(Scalar::from_uint(key, dest.layout.size), dest)?;
            }
            "TlsGetValue" => {
                let [key] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(u32) -> *_),
                    (link_name, abi, args),
                )?;
                let key = u128::from(ecx.read_scalar(key)?.to_u32()?);
                let active_thread = ecx.active_thread();
                let ptr = ecx.machine.tls.load_tls_key(key, active_thread)?;
                ecx.write_pointer(ptr, dest)?;
            }
            "TlsSetValue" => {
                let [key, new_ptr] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(u32, *_) -> winapi::BOOL),
                    (link_name, abi, args),
                )?;
                let key = u128::from(ecx.read_scalar(key)?.to_u32()?);
                let active_thread = ecx.active_thread();
                let new_data = ecx.read_pointer(new_ptr)?;
                ecx.machine.tls.store_tls_key(key, active_thread, new_data)?;

                // Return success (`1`).
                ecx.write_int(1, dest)?;
            }
            "TlsFree" => {
                let [key] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(u32) -> winapi::BOOL),
                    (link_name, abi, args),
                )?;
                let key = u128::from(ecx.read_scalar(key)?.to_u32()?);
                ecx.machine.tls.delete_tls_key(key)?;

                // Return success (`1`).
                ecx.write_int(1, dest)?;
            }

            // Fiber-local storage - similar to TLS but supports destructors.
            "FlsAlloc" => {
                // Create key and return it.
                let [dtor] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(winapi::PFLS_CALLBACK_FUNCTION) -> u32),
                    (link_name, abi, args),
                )?;
                let dtor = ecx.read_pointer(dtor)?;

                // Extract the function type out of the signature (that seems easier than constructing it ourselves).
                let dtor = if !ecx.ptr_is_null(dtor)? {
                    Some((
                        ecx.get_ptr_fn(dtor)?.as_instance()?,
                        ecx.machine.current_user_relevant_span(),
                    ))
                } else {
                    None
                };

                let key = ecx.machine.tls.create_tls_key(dtor, dest.layout.size)?;
                ecx.write_scalar(Scalar::from_uint(key, dest.layout.size), dest)?;
            }
            "FlsGetValue" => {
                let [key] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(u32) -> *_),
                    (link_name, abi, args),
                )?;
                let key = u128::from(ecx.read_scalar(key)?.to_u32()?);
                let active_thread = ecx.active_thread();
                let ptr = ecx.machine.tls.load_tls_key(key, active_thread)?;
                ecx.write_pointer(ptr, dest)?;
            }
            "FlsSetValue" => {
                let [key, new_ptr] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(u32, *_) -> winapi::BOOL),
                    (link_name, abi, args),
                )?;
                let key = u128::from(ecx.read_scalar(key)?.to_u32()?);
                let active_thread = ecx.active_thread();
                let new_data = ecx.read_pointer(new_ptr)?;
                ecx.machine.tls.store_tls_key(key, active_thread, new_data)?;

                // Return success (`1`).
                ecx.write_int(1, dest)?;
            }
            "FlsFree" => {
                let [key] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(u32) -> winapi::BOOL),
                    (link_name, abi, args),
                )?;
                let key = u128::from(ecx.read_scalar(key)?.to_u32()?);
                let tls_entry = ecx.machine.tls.delete_tls_key(key)?;

                // FIXME: We should run the destructor here *for all threads*. But that's non-trivial and std doesn't need it so we bail out with an "unsupported" error.
                if !tls_entry.data.is_empty() && tls_entry.dtor.is_some() {
                    throw_unsup_format!(
                        "calling `FlsFree` on a key with an associated dtor is not supported"
                    );
                }

                // Return success (`1`).
                ecx.write_int(1, dest)?;
            }
            "IsThreadAFiber" => {
                let [] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn() -> winapi::BOOL),
                    (link_name, abi, args),
                )?;

                // Return FALSE, as Miri does not support fibers.
                ecx.write_int(0, dest)?;
            }

            // Access to command-line arguments
            "GetCommandLineW" => {
                // FIXME: This does not have a direct test (#3179).
                let [] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn() -> *_),
                    (link_name, abi, args),
                )?;
                ecx.write_pointer(
                    ecx.machine.cmd_line.expect("machine must be initialized"),
                    dest,
                )?;
            }

            // Time related shims
            "GetSystemTimeAsFileTime" | "GetSystemTimePreciseAsFileTime" => {
                // FIXME: This does not have a direct test (#3179).
                let [filetime] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(*_) -> ()),
                    (link_name, abi, args),
                )?;
                ecx.GetSystemTimeAsFileTime(link_name.as_str(), filetime)?;
            }
            "QueryPerformanceCounter" => {
                // FIXME: This does not have a direct test (#3179).
                let [performance_count] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(*_) -> winapi::BOOL),
                    (link_name, abi, args),
                )?;
                let result = ecx.QueryPerformanceCounter(performance_count)?;
                ecx.write_scalar(result, dest)?;
            }
            "QueryPerformanceFrequency" => {
                // FIXME: This does not have a direct test (#3179).
                let [frequency] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(*_) -> winapi::BOOL),
                    (link_name, abi, args),
                )?;
                let result = ecx.QueryPerformanceFrequency(frequency)?;
                ecx.write_scalar(result, dest)?;
            }
            "Sleep" => {
                // FIXME: This does not have a direct test (#3179).
                let [timeout] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(u32) -> ()),
                    (link_name, abi, args),
                )?;

                ecx.Sleep(timeout)?;
            }
            "CreateWaitableTimerExW" => {
                // FIXME: This does not have a direct test (#3179).
                let [attributes, name, flags, access] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(*_, *_, u32, u32) -> winapi::HANDLE),
                    (link_name, abi, args),
                )?;
                ecx.read_pointer(attributes)?;
                ecx.read_pointer(name)?;
                ecx.read_scalar(flags)?.to_u32()?;
                ecx.read_scalar(access)?.to_u32()?;
                // Unimplemented. Always return failure.
                let not_supported = ecx.eval_windows("c", "ERROR_NOT_SUPPORTED");
                ecx.set_last_error(not_supported)?;
                ecx.write_null(dest)?;
            }

            // Synchronization primitives
            "InitOnceBeginInitialize" => {
                let [ptr, flags, pending, context] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(*_, u32, *_, *_) -> winapi::BOOL),
                    (link_name, abi, args),
                )?;
                ecx.InitOnceBeginInitialize(ptr, flags, pending, context, dest)?;
            }
            "InitOnceComplete" => {
                let [ptr, flags, context] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(*_, u32, *_) -> winapi::BOOL),
                    (link_name, abi, args),
                )?;
                let result = ecx.InitOnceComplete(ptr, flags, context)?;
                ecx.write_scalar(result, dest)?;
            }
            "WaitOnAddress" => {
                // FIXME: This does not have a direct test (#3179).
                let [ptr_op, compare_op, size_op, timeout_op] = &ecx.check_shim_sig(
                    // First pointer is volatile
                    shim_sig!(extern "system" fn(*_, *_, usize, u32) -> winapi::BOOL),
                    (link_name, abi, args),
                )?;

                ecx.WaitOnAddress(ptr_op, compare_op, size_op, timeout_op, dest)?;
            }
            "WakeByAddressSingle" => {
                // FIXME: This does not have a direct test (#3179).
                let [ptr_op] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(*_) -> ()),
                    (link_name, abi, args),
                )?;

                ecx.WakeByAddressSingle(ptr_op)?;
            }
            "WakeByAddressAll" => {
                // FIXME: This does not have a direct test (#3179).
                let [ptr_op] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(*_) -> ()),
                    (link_name, abi, args),
                )?;

                ecx.WakeByAddressAll(ptr_op)?;
            }

            // Dynamic symbol loading
            "GetProcAddress" => {
                // FIXME: This does not have a direct test (#3179).
                let [module, proc_name] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(winapi::HMODULE, *_) -> winapi::FARPROC),
                    (link_name, abi, args),
                )?;
                ecx.read_target_isize(module)?; // FIXME validate the module!
                let name = ecx.read_c_str(ecx.read_pointer(proc_name)?)?;
                if let Ok(name) = str::from_utf8(name)
                    && is_dyn_sym(name)
                {
                    let ptr = ecx.fn_ptr(FnVal::Other(DynSym::from_str(name)));
                    ecx.write_pointer(ptr, dest)?;
                } else {
                    ecx.write_null(dest)?;
                }
            }

            // Threading
            "CreateThread" => {
                let [security, stacksize, start, arg, flags, thread] = &ecx.check_shim_sig(
                    shim_sig!(
                        extern "system" fn(
                            *_,
                            usize,
                            *_,
                            *_,
                            u32,
                            *_,
                        ) -> winapi::HANDLE
                    ),
                    (link_name, abi, args),
                )?;

                let thread_id = ecx.CreateThread(security, stacksize, start, arg, flags, thread)?;

                ecx.write_scalar(Handle::Thread(thread_id).to_scalar(ecx), dest)?;
            }
            "WaitForSingleObject" => {
                let [handle, timeout] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(winapi::HANDLE, u32) -> u32),
                    (link_name, abi, args),
                )?;

                ecx.WaitForSingleObject(handle, timeout, dest)?;
            }
            "GetCurrentProcess" => {
                let [] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn() -> winapi::HANDLE),
                    (link_name, abi, args),
                )?;

                ecx.write_scalar(
                    Handle::Pseudo(PseudoHandle::CurrentProcess).to_scalar(ecx),
                    dest,
                )?;
            }
            "GetCurrentThread" => {
                let [] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn() -> winapi::HANDLE),
                    (link_name, abi, args),
                )?;

                ecx.write_scalar(Handle::Pseudo(PseudoHandle::CurrentThread).to_scalar(ecx), dest)?;
            }
            "SetThreadDescription" => {
                let [handle, name] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(winapi::HANDLE, *_) -> i32),
                    (link_name, abi, args),
                )?;

                let handle = ecx.read_handle(handle, "SetThreadDescription")?;
                let name = ecx.read_wide_str(ecx.read_pointer(name)?)?;

                let thread = match handle {
                    Handle::Thread(thread) => thread,
                    Handle::Pseudo(PseudoHandle::CurrentThread) => ecx.active_thread(),
                    _ => ecx.invalid_handle("SetThreadDescription")?,
                };
                // FIXME: use non-lossy conversion
                ecx.set_thread_name(thread, String::from_utf16_lossy(&name).into_bytes());
                ecx.write_scalar(Scalar::from_u32(0), dest)?;
            }
            "GetThreadDescription" => {
                let [handle, name_ptr] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(winapi::HANDLE, *_) -> i32),
                    (link_name, abi, args),
                )?;

                let handle = ecx.read_handle(handle, "GetThreadDescription")?;
                let name_ptr = ecx.deref_pointer_as(name_ptr, ecx.machine.layouts.unit_ptr_mut)?; // the pointer where we should store the ptr to the name

                let thread = match handle {
                    Handle::Thread(thread) => thread,
                    Handle::Pseudo(PseudoHandle::CurrentThread) => ecx.active_thread(),
                    _ => ecx.invalid_handle("GetThreadDescription")?,
                };
                // Looks like the default thread name is empty.
                let name = ecx.get_thread_name(thread).unwrap_or(b"").to_owned();
                let name = ecx.alloc_os_str_as_wide_str(
                    bytes_to_os_str(&name)?,
                    MiriMemoryKind::WinLocal.into(),
                )?;
                let name = Scalar::from_maybe_pointer(name, ecx);
                let res = Scalar::from_u32(0);

                ecx.write_scalar(name, &name_ptr)?;
                ecx.write_scalar(res, dest)?;
            }
            "GetThreadId" => {
                let [handle] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(winapi::HANDLE) -> u32),
                    (link_name, abi, args),
                )?;
                let handle = ecx.read_handle(handle, "GetThreadId")?;
                let thread = match handle {
                    Handle::Thread(thread) => thread,
                    Handle::Pseudo(PseudoHandle::CurrentThread) => ecx.active_thread(),
                    _ => ecx.invalid_handle("GetThreadDescription")?,
                };
                ecx.write_scalar(Scalar::from_u32(thread.to_u32()), dest)?;
            }
            "GetCurrentThreadId" => {
                let [] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn() -> u32),
                    (link_name, abi, args),
                )?;
                ecx.write_scalar(Scalar::from_u32(ecx.active_thread().to_u32()), dest)?;
            }

            // Miscellaneous
            "ExitProcess" => {
                // FIXME: This does not have a direct test (#3179).
                let [code] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(u32) -> ()),
                    (link_name, abi, args),
                )?;
                // Windows technically uses u32, but we unify everything to a Unix-style i32.
                let exit_code = ecx.read_scalar(code)?.to_i32()?;
                // Initialize the at-exit state machine.
                ecx.replace_active_thread_stack({
                    let mut state_machine =
                        ProgramExitHandler::new(exit_code, ExitType::ExitCalled);
                    Box::new(move |m| state_machine.on_stack_empty(m))
                });
                // We "jumped" to the exit handling.
                return interp_ok(EmulateItemResult::AlreadyJumped);
            }
            "SystemFunction036" => {
                // used by getrandom 0.1
                // This is really 'RtlGenRandom'.
                let [ptr, len] = &ecx.check_shim_sig(
                    // Returns winapi::BOOLEAN, which is a byte
                    shim_sig!(extern "system" fn(*_, u32) -> u8),
                    (link_name, abi, args),
                )?;
                let ptr = ecx.read_pointer(ptr)?;
                let len = ecx.read_scalar(len)?.to_u32()?;
                ecx.gen_random(ptr, len.into())?;
                ecx.write_scalar(Scalar::from_bool(true), dest)?;
            }
            "ProcessPrng" => {
                // used by `std`
                let [ptr, len] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(*_, usize) -> winapi::BOOL),
                    (link_name, abi, args),
                )?;
                let ptr = ecx.read_pointer(ptr)?;
                let len = ecx.read_target_usize(len)?;
                ecx.gen_random(ptr, len)?;
                ecx.write_int(1, dest)?;
            }
            "BCryptGenRandom" => {
                // used by getrandom 0.2
                let [algorithm, ptr, len, flags] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(*_, *_, u32, u32) -> i32),
                    (link_name, abi, args),
                )?;
                let algorithm = ecx.read_scalar(algorithm)?;
                let algorithm = algorithm.to_target_usize(ecx)?;
                let ptr = ecx.read_pointer(ptr)?;
                let len = ecx.read_scalar(len)?.to_u32()?;
                let flags = ecx.read_scalar(flags)?.to_u32()?;
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
                ecx.gen_random(ptr, len.into())?;
                ecx.write_null(dest)?; // STATUS_SUCCESS
            }
            "GetConsoleScreenBufferInfo" => {
                // FIXME: This does not have a direct test (#3179).
                // `term` needs this, so we fake it.
                let [console, buffer_info] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(winapi::HANDLE, *_) -> winapi::BOOL),
                    (link_name, abi, args),
                )?;
                ecx.read_target_isize(console)?;
                ecx.read_pointer(buffer_info)?;
                // Indicate an error.
                ecx.set_last_error(io::ErrorKind::Unsupported)?;
                ecx.write_null(dest)?;
            }
            "GetStdHandle" => {
                // FIXME: This does not have a direct test (#3179).
                let [which] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(u32) -> winapi::HANDLE),
                    (link_name, abi, args),
                )?;
                let res = ecx.GetStdHandle(which)?;
                ecx.write_scalar(res, dest)?;
            }
            "DuplicateHandle" => {
                let [src_proc, src_handle, target_proc, target_handle, access, inherit, options] =
                    &ecx.check_shim_sig(
                        shim_sig!(
                            extern "system" fn(
                                winapi::HANDLE,
                                winapi::HANDLE,
                                winapi::HANDLE,
                                *_,
                                u32,
                                winapi::BOOL,
                                u32,
                            ) -> winapi::BOOL
                        ),
                        (link_name, abi, args),
                    )?;
                let res = ecx.DuplicateHandle(
                    src_proc,
                    src_handle,
                    target_proc,
                    target_handle,
                    access,
                    inherit,
                    options,
                )?;
                ecx.write_scalar(res, dest)?;
            }
            "CloseHandle" => {
                let [handle] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(winapi::HANDLE) -> winapi::BOOL),
                    (link_name, abi, args),
                )?;

                let ret = ecx.CloseHandle(handle)?;

                ecx.write_scalar(ret, dest)?;
            }
            "GetModuleFileNameW" => {
                // FIXME: This does not have a direct test (#3179).
                let [handle, filename, size] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(winapi::HMODULE, *_, u32) -> u32),
                    (link_name, abi, args),
                )?;
                ecx.check_no_isolation("`GetModuleFileNameW`")?;

                let handle = ecx.read_handle(handle, "GetModuleFileNameW")?;
                let filename = ecx.read_pointer(filename)?;
                let size = ecx.read_scalar(size)?.to_u32()?;

                if handle != Handle::Null {
                    throw_unsup_format!("`GetModuleFileNameW` only supports the NULL handle");
                }

                let Some(path) = ecx.machine.current_exe.clone() else {
                    throw_unsup_format!("GetModuleFileNameW only works when Miri is run on a file")
                };
                let (all_written, size_needed) =
                    ecx.write_path_to_wide_str_truncated(&path, filename, size.into())?;

                if all_written {
                    // If the function succeeds, the return value is the length of the string that
                    // is copied to the buffer, in characters, not including the terminating null
                    // character.
                    ecx.write_int(size_needed.strict_sub(1), dest)?;
                } else {
                    // If the buffer is too small to hold the module name, the string is truncated
                    // to nSize characters including the terminating null character, the function
                    // returns nSize, and the function sets the last error to
                    // ERROR_INSUFFICIENT_BUFFER.
                    ecx.write_int(size, dest)?;
                    let insufficient_buffer = ecx.eval_windows("c", "ERROR_INSUFFICIENT_BUFFER");
                    ecx.set_last_error(insufficient_buffer)?;
                }
            }
            "FormatMessageW" => {
                // FIXME: This does not have a direct test (#3179).
                let [flags, module, message_id, language_id, buffer, size, arguments] = &ecx
                    .check_shim_sig(
                        shim_sig!(
                            extern "system" fn(u32, *_, u32, u32, *_, u32, *_) -> u32
                        ),
                        (link_name, abi, args),
                    )?;

                let flags = ecx.read_scalar(flags)?.to_u32()?;
                let _module = ecx.read_pointer(module)?; // seems to contain a module name
                let message_id = ecx.read_scalar(message_id)?;
                let _language_id = ecx.read_scalar(language_id)?.to_u32()?;
                let buffer = ecx.read_pointer(buffer)?;
                let size = ecx.read_scalar(size)?.to_u32()?;
                let _arguments = ecx.read_pointer(arguments)?;

                // We only support `FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS`
                // This also means `arguments` can be ignored.
                if flags != 4096u32 | 512u32 {
                    throw_unsup_format!("FormatMessageW: unsupported flags {flags:#x}");
                }

                let error = ecx.try_errnum_to_io_error(message_id)?;
                let formatted = match error {
                    Some(err) => format!("{err}"),
                    None => format!("<unknown error in FormatMessageW: {message_id}>"),
                };
                let (complete, length) =
                    ecx.write_os_str_to_wide_str(OsStr::new(&formatted), buffer, size.into())?;
                if !complete {
                    // The API docs don't say what happens when the buffer is not big enough...
                    // Let's just bail.
                    throw_unsup_format!("FormatMessageW: buffer not big enough");
                }
                // The return value is the number of characters stored *excluding* the null terminator.
                ecx.write_int(length.strict_sub(1), dest)?;
            }

            "_Unwind_RaiseException" => {
                // This is not formally part of POSIX, but it is very wide-spread on POSIX systems.
                // It was originally specified as part of the Itanium C++ ABI:
                // https://itanium-cxx-abi.github.io/cxx-abi/abi-eh.html#base-throw.
                // MinGW implements _Unwind_RaiseException on top of SEH exceptions.
                if ecx.tcx.sess.target.env != Env::Gnu {
                    throw_unsup_format!(
                        "`_Unwind_RaiseException` is not supported on non-MinGW Windows",
                    );
                }
                // This function looks and behaves exactly like miri_start_unwind.
                let [payload] = &ecx.check_shim_sig(
                    // Look up the return type via `panic_unwind::`, not via `unwind::`, as
                    // the latter it not always unique.
                    shim_sig!(extern "C" fn(*_) -> panic_unwind::imp::uw::_Unwind_Reason_Code),
                    (link_name, abi, args),
                )?;
                ecx.handle_miri_start_unwind(payload)?;
                return interp_ok(EmulateItemResult::NeedsUnwind);
            }

            // Incomplete shims that we "stub out" just to get pre-main initialization code to work.
            // These shims are enabled only when the caller is in the standard library.
            "NtQueryVolumeInformationFile" if ecx.frame_in_std() => {
                let [_handle, _iosb, _info, _length, _class] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(winapi::HANDLE, *_, *_, u32, i32) -> i32),
                    (link_name, abi, args),
                )?;
                // Return an error to make std think that nothing is a console.
                // This means the normal NtWriteFile is used for stdout/stderr, greatly simplifying our lives.
                let not_implemented = ecx.eval_windows("c", "STATUS_NOT_IMPLEMENTED");
                ecx.write_scalar(not_implemented, dest)?;
            }
            "GetProcessHeap" if ecx.frame_in_std() => {
                let [] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn() -> winapi::HANDLE),
                    (link_name, abi, args),
                )?;
                // Just fake a HANDLE
                // It's fine to not use the Handle type here because its a stub
                ecx.write_int(1, dest)?;
            }
            "GetModuleHandleA" if ecx.frame_in_std() => {
                let [_module_name] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(*_) -> winapi::HMODULE),
                    (link_name, abi, args),
                )?;
                // We need to return something non-null here to make `compat_fn!` work.
                ecx.write_int(1, dest)?;
            }
            "SetConsoleTextAttribute" if ecx.frame_in_std() => {
                let [_console_output, _attribute] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(winapi::HANDLE, u16) -> winapi::BOOL),
                    (link_name, abi, args),
                )?;
                // Pretend these does not exist / nothing happened, by returning zero.
                ecx.write_null(dest)?;
            }
            "AddVectoredExceptionHandler" if ecx.frame_in_std() => {
                let [_first, _handler] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(u32, *_) -> *_),
                    (link_name, abi, args),
                )?;
                // Any non zero value works for the stdlib. This is just used for stack overflows anyway.
                ecx.write_int(1, dest)?;
            }
            "SetThreadStackGuarantee" if ecx.frame_in_std() => {
                let [_stack_size_in_bytes] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(*_) -> winapi::BOOL),
                    (link_name, abi, args),
                )?;
                // Any non zero value works for the stdlib. This is just used for stack overflows anyway.
                ecx.write_int(1, dest)?;
            }
            "SwitchToThread" if ecx.frame_in_std() => {
                // this is only callable from std because we know that std ignores the return value
                let [] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn() -> winapi::BOOL),
                    (link_name, abi, args),
                )?;

                ecx.yield_active_thread();

                // FIXME: this should return a nonzero value if this call does result in switching to another thread.
                ecx.write_null(dest)?;
            }
            "GetFinalPathNameByHandleW" if ecx.frame_in_std() => {
                // Std uses this when debug-printing `File`. We stub it out by returning an error.
                let [_file, _buf, _bufsz, _flags] = &ecx.check_shim_sig(
                    shim_sig!(extern "system" fn(winapi::HANDLE, *_, u32, u32) -> u32),
                    (link_name, abi, args),
                )?;

                ecx.set_last_error(IoError::WindowsError("ERROR_NOT_SUPPORTED"))?;
                ecx.write_null(dest)?;
            }

            // this is only callable from std, which always asks for `ALL_PROCESSOR_GROUPS`
            "GetActiveProcessorCount" if this.frame_in_std() => {
                // FIXME: This does not have a direct test (#3179).
                let [group_number] = this.check_shim_sig(
                    shim_sig!(extern "system" fn(u16) -> u32),
                    link_name,
                    abi,
                    args,
                )?;
                const ALL_PROCESSOR_GROUPS: u16 = 0xffff;
                if this.read_scalar(group_number)?.to_u16()? != ALL_PROCESSOR_GROUPS {
                    throw_unsup_format!(
                        "GetActiveProcessorCount: only ALL_PROCESSOR_GROUPS is supported"
                    );
                }
                // Miri does not model processor groups, so report every CPU.
                this.write_scalar(Scalar::from_u32(this.machine.num_cpus), dest)?;
            }
            // this is only callable from std, which ignores the group array and the return value
            "GetProcessGroupAffinity" if this.frame_in_std() => {
                // FIXME: This does not have a direct test (#3179).
                let [_handle, group_count, _group_array] = this.check_shim_sig(
                    shim_sig!(extern "system" fn(winapi::HANDLE, *mut _, *mut _) -> winapi::BOOL),
                    link_name,
                    abi,
                    args,
                )?;
                // Report more than one group so `available_parallelism` takes the
                // `GetActiveProcessorCount(ALL_PROCESSOR_GROUPS)` path, giving Miri
                // coverage of the multi-group branch.
                let group_count = this.deref_pointer_as(group_count, this.machine.layouts.u16)?;
                this.write_scalar(Scalar::from_u16(2), &group_count)?;
                this.write_scalar(Scalar::from_i32(1), dest)?; // TRUE
            }

            _ => return interp_ok(EmulateItemResult::NotSupported),
        }

        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
