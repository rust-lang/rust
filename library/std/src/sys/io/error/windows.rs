use crate::sys::pal::{api, c};
use crate::{io, ptr};

pub fn errno() -> i32 {
    api::get_last_error().code as i32
}

#[inline]
pub fn is_interrupted(_errno: i32) -> bool {
    false
}

pub fn decode_error_kind(errno: i32) -> io::ErrorKind {
    use io::ErrorKind::*;

    match errno as u32 {
        c::ERROR_ACCESS_DENIED => return PermissionDenied,
        c::ERROR_ALREADY_EXISTS => return AlreadyExists,
        c::ERROR_FILE_EXISTS => return AlreadyExists,
        c::ERROR_BROKEN_PIPE => return BrokenPipe,
        c::ERROR_FILE_NOT_FOUND
        | c::ERROR_PATH_NOT_FOUND
        | c::ERROR_INVALID_DRIVE
        | c::ERROR_BAD_NETPATH
        | c::ERROR_BAD_NET_NAME => return NotFound,
        c::ERROR_NO_DATA => return BrokenPipe,
        c::ERROR_INVALID_NAME | c::ERROR_BAD_PATHNAME => return InvalidFilename,
        c::ERROR_INVALID_PARAMETER => return InvalidInput,
        c::ERROR_NOT_ENOUGH_MEMORY | c::ERROR_OUTOFMEMORY => return OutOfMemory,
        c::ERROR_SEM_TIMEOUT
        | c::WAIT_TIMEOUT
        | c::ERROR_DRIVER_CANCEL_TIMEOUT
        | c::ERROR_OPERATION_ABORTED
        | c::ERROR_SERVICE_REQUEST_TIMEOUT
        | c::ERROR_COUNTER_TIMEOUT
        | c::ERROR_TIMEOUT
        | c::ERROR_RESOURCE_CALL_TIMED_OUT
        | c::ERROR_CTX_MODEM_RESPONSE_TIMEOUT
        | c::ERROR_CTX_CLIENT_QUERY_TIMEOUT
        | c::FRS_ERR_SYSVOL_POPULATE_TIMEOUT
        | c::ERROR_DS_TIMELIMIT_EXCEEDED
        | c::DNS_ERROR_RECORD_TIMED_OUT
        | c::ERROR_IPSEC_IKE_TIMED_OUT
        | c::ERROR_RUNLEVEL_SWITCH_TIMEOUT
        | c::ERROR_RUNLEVEL_SWITCH_AGENT_TIMEOUT => return TimedOut,
        c::ERROR_CALL_NOT_IMPLEMENTED => return Unsupported,
        c::ERROR_HOST_UNREACHABLE => return HostUnreachable,
        c::ERROR_NETWORK_UNREACHABLE => return NetworkUnreachable,
        c::ERROR_DIRECTORY => return NotADirectory,
        c::ERROR_DIRECTORY_NOT_SUPPORTED => return IsADirectory,
        c::ERROR_DIR_NOT_EMPTY => return DirectoryNotEmpty,
        c::ERROR_WRITE_PROTECT => return ReadOnlyFilesystem,
        c::ERROR_DISK_FULL | c::ERROR_HANDLE_DISK_FULL => return StorageFull,
        c::ERROR_SEEK_ON_DEVICE => return NotSeekable,
        c::ERROR_DISK_QUOTA_EXCEEDED => return QuotaExceeded,
        c::ERROR_FILE_TOO_LARGE => return FileTooLarge,
        c::ERROR_BUSY => return ResourceBusy,
        c::ERROR_POSSIBLE_DEADLOCK => return Deadlock,
        c::ERROR_NOT_SAME_DEVICE => return CrossesDevices,
        c::ERROR_TOO_MANY_LINKS => return TooManyLinks,
        c::ERROR_FILENAME_EXCED_RANGE => return InvalidFilename,
        c::ERROR_CANT_RESOLVE_FILENAME => return FilesystemLoop,
        _ => {}
    }

    match errno {
        c::WSAEACCES => PermissionDenied,
        c::WSAEADDRINUSE => AddrInUse,
        c::WSAEADDRNOTAVAIL => AddrNotAvailable,
        c::WSAECONNABORTED => ConnectionAborted,
        c::WSAECONNREFUSED => ConnectionRefused,
        c::WSAECONNRESET => ConnectionReset,
        c::WSAEINVAL => InvalidInput,
        c::WSAENOTCONN => NotConnected,
        c::WSAEWOULDBLOCK => WouldBlock,
        c::WSAETIMEDOUT => TimedOut,
        c::WSAEHOSTUNREACH => HostUnreachable,
        c::WSAENETDOWN => NetworkDown,
        c::WSAENETUNREACH => NetworkUnreachable,
        c::WSAEDQUOT => QuotaExceeded,

        _ => Uncategorized,
    }
}

/// Gets a detailed string description for the given error number.
pub fn error_string(mut errnum: i32) -> String {
    let mut buf = [0 as c::WCHAR; 2048];

    unsafe {
        let mut module = ptr::null_mut();
        let mut flags = 0;

        // NTSTATUS errors may be encoded as HRESULT, which may returned from
        // GetLastError. For more information about Windows error codes, see
        // `[MS-ERREF]`: https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-erref/0642cb2f-2075-4469-918c-4441e69c548a
        if (errnum & c::FACILITY_NT_BIT as i32) != 0 {
            // format according to https://support.microsoft.com/en-us/help/259693
            const NTDLL_DLL: &[u16] = &[
                'N' as _, 'T' as _, 'D' as _, 'L' as _, 'L' as _, '.' as _, 'D' as _, 'L' as _,
                'L' as _, 0,
            ];
            module = c::GetModuleHandleW(NTDLL_DLL.as_ptr());

            if !module.is_null() {
                errnum ^= c::FACILITY_NT_BIT as i32;
                flags = c::FORMAT_MESSAGE_FROM_HMODULE;
            }
        }

        let res = c::FormatMessageW(
            flags | c::FORMAT_MESSAGE_FROM_SYSTEM | c::FORMAT_MESSAGE_IGNORE_INSERTS,
            module,
            errnum as u32,
            0,
            buf.as_mut_ptr(),
            buf.len() as u32,
            ptr::null(),
        ) as usize;
        if res == 0 {
            // Sometimes FormatMessageW can fail e.g., system doesn't like 0 as langId,
            let fm_err = errno();
            return format!("OS Error {errnum} (FormatMessageW() returned error {fm_err})");
        }

        match String::from_utf16(&buf[..res]) {
            Ok(mut msg) => {
                // Trim trailing CRLF inserted by FormatMessageW
                let len = msg.trim_end().len();
                msg.truncate(len);
                msg
            }
            Err(..) => format!(
                "OS Error {} (FormatMessageW() returned \
                 invalid UTF-16)",
                errnum
            ),
        }
    }
}
