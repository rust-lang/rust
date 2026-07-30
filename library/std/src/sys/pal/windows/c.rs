//! C definitions used by libnative that don't belong in liblibc

#![allow(nonstandard_style)]
#![cfg_attr(test, allow(dead_code))]
#![unstable(issue = "none", feature = "windows_c")]
#![allow(clippy::style)]

use core::ffi::{CStr, c_int, c_uint, c_ulong, c_ushort, c_void};
use core::ptr;

#[allow(unused)]
mod windows_sys;
pub use windows_sys::*;

pub type WCHAR = u16;

pub const INVALID_HANDLE_VALUE: HANDLE = ::core::ptr::without_provenance_mut(-1i32 as _);

// https://learn.microsoft.com/en-us/cpp/c-runtime-library/exit-success-exit-failure?view=msvc-170
pub const EXIT_SUCCESS: u32 = 0;
pub const EXIT_FAILURE: u32 = 1;

// For use in the below macro only.
// Asserting on this produces a better message in a macro then `fn as_u32(i: i32) -> u32` does.
const fn is_i32<T: crate::any::Any>(_i: &T) -> bool {
    crate::intrinsics::type_id_eq(crate::any::TypeId::of::<T>(), crate::any::TypeId::of::<i32>())
}

macro_rules! as_u32 {
    ($($(#[$meta:meta])*$ident:ident),+,) => {
        $(
            $(#[$meta])*
            pub const $ident: u32 = {
                assert!(is_i32(&windows_sys::$ident));
                windows_sys::$ident as u32
            };
        )+
    };
}

macro_rules! flags {
    ($($tt:tt)*) => {
        as_u32!($($tt)*);
    };
}

macro_rules! errors {
    ($($tt:tt)*) => {
        as_u32!($($tt)*);
    };
}

flags! {
    // dwShareMode -- CreateFile
    FILE_SHARE_DELETE,
    FILE_SHARE_READ,
    FILE_SHARE_WRITE,

    // CreateOptions (ULONG) -- NtCreateFile
    FILE_DIRECTORY_FILE,
    FILE_SYNCHRONOUS_IO_NONALERT,
    FILE_NON_DIRECTORY_FILE,
    FILE_OPEN_REPARSE_POINT,

    // dwDesiredAccess -- many, many functions
    DELETE,
    SYNCHRONIZE,
    //GENERIC_READ,
    GENERIC_WRITE,
    FILE_GENERIC_WRITE,
    FILE_LIST_DIRECTORY,
    FILE_WRITE_DATA,
    FILE_READ_ATTRIBUTES,
    FILE_WRITE_ATTRIBUTES,
    TIMER_ALL_ACCESS,
    #[cfg(target_vendor = "win7")]
    TOKEN_READ,

    // dwOptions -- DuplicateHandle
    DUPLICATE_SAME_ACCESS,

    // dwFlags -- GetModuleHandleEx
    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
    GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,

    // dwSecurityQosFlags -- CreateFile
    SECURITY_SQOS_PRESENT,

    // dwFlags -- LockFileEx
    LOCKFILE_EXCLUSIVE_LOCK,
    LOCKFILE_FAIL_IMMEDIATELY,

    // dwFileFlags -- CreateFile
    FILE_FLAG_BACKUP_SEMANTICS,
    FILE_FLAG_OPEN_REPARSE_POINT,
    FILE_FLAG_POSIX_SEMANTICS,

    // dwFileAttributes -- CreateFile
    FILE_ATTRIBUTE_DIRECTORY,
    FILE_ATTRIBUTE_NORMAL,
    FILE_ATTRIBUTE_READONLY,
    FILE_ATTRIBUTE_REPARSE_POINT,

    // dwFlags -- WSASocket
    WSA_FLAG_NO_HANDLE_INHERIT,
    WSA_FLAG_OVERLAPPED,

    // Flags (ULONG) -- FILE_RENAME_INFORMATION
    FILE_RENAME_FLAG_REPLACE_IF_EXISTS,
    FILE_RENAME_FLAG_POSIX_SEMANTICS,

    // Attributes (ULONG) -- OBJECT_ATTRIBUTES
    OBJ_INHERIT,
    OBJ_DONT_REPARSE,

    // Flags (DWORD) -- FILE_DISPOSITION_INFO_EX
    FILE_DISPOSITION_FLAG_DELETE,
    FILE_DISPOSITION_FLAG_POSIX_SEMANTICS,
    FILE_DISPOSITION_FLAG_IGNORE_READONLY_ATTRIBUTE,

    // Flags (ULONG) -- SymbolicLinkReparseBuffer
    SYMLINK_FLAG_RELATIVE,

    // dwFlags -- CreateSymbolicLink
    SYMBOLIC_LINK_FLAG_DIRECTORY,
    SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE,

    // dwCreationFlag -- CreateProcess
    DETACHED_PROCESS,
    CREATE_NEW_PROCESS_GROUP,
    CREATE_UNICODE_ENVIRONMENT,
    EXTENDED_STARTUPINFO_PRESENT,

    // dwFlags -- STARTUPINFO
    STARTF_USESTDHANDLES,
    STARTF_USESHOWWINDOW,
    STARTF_UNTRUSTEDSOURCE,
    STARTF_FORCEONFEEDBACK,
    STARTF_FORCEOFFFEEDBACK,
    STARTF_RUNFULLSCREEN,

    // dwFlags -- MoveFileEx
    MOVEFILE_REPLACE_EXISTING,

    // dwFlags -- GetFinalPathNameByHandle
    VOLUME_NAME_DOS,

    // dwFlags -- FormatMessageW
    FORMAT_MESSAGE_FROM_SYSTEM,
    FORMAT_MESSAGE_IGNORE_INSERTS,
    FORMAT_MESSAGE_FROM_HMODULE,

    // dwFlags -- SetHandleInformation
    #[cfg(not(target_vendor = "uwp"))]
    HANDLE_FLAG_INHERIT,

    // dwCreationFlags -- CreateThread
    STACK_SIZE_PARAM_IS_A_RESERVATION,

    // dwFlags -- WideCharToMultiByte
    WC_ERR_INVALID_CHARS,

    // dwFlags -- MultiByteToWideChar
    MB_ERR_INVALID_CHARS,

    // dwFlags -- CreateWaitableTimerExW
    CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
}

as_u32! {
    // dwCreationDisposition -- CreateFile
    CREATE_NEW,
    OPEN_ALWAYS,
    OPEN_EXISTING,
    TRUNCATE_EXISTING,

    // CreateDisposition (ULONG) -- NtCreateFile
    FILE_OPEN,
    FILE_CREATE,
    FILE_OPEN_IF,
    FILE_OVERWRITE,
    FILE_OVERWRITE_IF,

    // NamedPipeType (ULONG) -- NtCreateNamedPipeFile
    FILE_PIPE_BYTE_STREAM_TYPE,

    // ReadMode (ULONG) -- NtCreateNamedPipeFile
    FILE_PIPE_BYTE_STREAM_MODE,

    // CompletionMode (ULONG) -- NtCreateNamedPipeFile
    FILE_PIPE_QUEUE_OPERATION,

    // dwMoveMethod -- SetFilePointerEx
    FILE_BEGIN,
    FILE_END,
    FILE_CURRENT,

    // dwIoControlCode -- DeviceIoControl
    FSCTL_GET_REPARSE_POINT,
    FSCTL_SET_REPARSE_POINT,

    // GetFileType return value (DWORD)
    FILE_TYPE_PIPE,

    // GetACP return value (UINT)
    CP_UTF8,

    // WaitForSingleObject return value (DWORD)
    WAIT_OBJECT_0,
    //WAIT_FAILED,

    // LPPROGRESS_ROUTINE return value (DWORD)
    PROGRESS_CONTINUE,
}

// ADDRESS_FAMILY
pub const AF_INET: ADDRESS_FAMILY = windows_sys::AF_INET as _;
pub const AF_INET6: ADDRESS_FAMILY = windows_sys::AF_INET6 as _;
pub const AF_UNIX: ADDRESS_FAMILY = windows_sys::AF_UNIX as _;

errors! {
    ERROR_SUCCESS,
    ERROR_ACCESS_DENIED,
    ERROR_ALREADY_EXISTS,
    ERROR_BAD_NETPATH,
    ERROR_BAD_NET_NAME,
    ERROR_CANT_ACCESS_FILE,
    ERROR_DELETE_PENDING,
    ERROR_DIRECTORY,
    ERROR_DIR_NOT_EMPTY,
    ERROR_FILE_NOT_FOUND,
    ERROR_INSUFFICIENT_BUFFER,
    ERROR_INVALID_FUNCTION,
    ERROR_INVALID_HANDLE,
    ERROR_INVALID_PARAMETER,
    ERROR_NOT_FOUND,
    ERROR_NOT_SUPPORTED,
    ERROR_NO_MORE_FILES,
    ERROR_OPERATION_ABORTED,
    ERROR_PATH_NOT_FOUND,
    ERROR_SHARING_VIOLATION,
    ERROR_TIMEOUT,
    ERROR_FILE_EXISTS,
    ERROR_BROKEN_PIPE,
    ERROR_INVALID_DRIVE,
    ERROR_NO_DATA,
    ERROR_INVALID_NAME,
    ERROR_BAD_PATHNAME,
    ERROR_NOT_ENOUGH_MEMORY,
    ERROR_OUTOFMEMORY,
    ERROR_SEM_TIMEOUT,
    ERROR_DRIVER_CANCEL_TIMEOUT,
    ERROR_SERVICE_REQUEST_TIMEOUT,
    ERROR_COUNTER_TIMEOUT,
    ERROR_RESOURCE_CALL_TIMED_OUT,
    ERROR_CTX_MODEM_RESPONSE_TIMEOUT,
    ERROR_CTX_CLIENT_QUERY_TIMEOUT,
    ERROR_DS_TIMELIMIT_EXCEEDED,
    DNS_ERROR_RECORD_TIMED_OUT,
    ERROR_IPSEC_IKE_TIMED_OUT,
    ERROR_RUNLEVEL_SWITCH_TIMEOUT,
    ERROR_RUNLEVEL_SWITCH_AGENT_TIMEOUT,
    ERROR_CALL_NOT_IMPLEMENTED,
    ERROR_HOST_UNREACHABLE,
    ERROR_NETWORK_UNREACHABLE,
    ERROR_DIRECTORY_NOT_SUPPORTED,
    ERROR_WRITE_PROTECT,
    ERROR_DISK_FULL,
    ERROR_HANDLE_DISK_FULL,
    ERROR_SEEK_ON_DEVICE,
    ERROR_DISK_QUOTA_EXCEEDED,
    ERROR_FILE_TOO_LARGE,
    ERROR_BUSY,
    ERROR_POSSIBLE_DEADLOCK,
    ERROR_NOT_SAME_DEVICE,
    ERROR_TOO_MANY_LINKS,
    ERROR_TOO_MANY_OPEN_FILES,
    ERROR_FILENAME_EXCED_RANGE,
    ERROR_CANT_RESOLVE_FILENAME,
    ERROR_IO_DEVICE,
    WAIT_TIMEOUT,
}

#[cfg(target_vendor = "win7")]
pub const CONDITION_VARIABLE_INIT: CONDITION_VARIABLE = CONDITION_VARIABLE { Ptr: ptr::null_mut() };
#[cfg(target_vendor = "win7")]
pub const SRWLOCK_INIT: SRWLOCK = SRWLOCK { Ptr: ptr::null_mut() };
#[cfg(not(target_thread_local))]
pub const INIT_ONCE_STATIC_INIT: INIT_ONCE = INIT_ONCE { Ptr: ptr::null_mut() };

// Equivalent to the `NT_SUCCESS` C preprocessor macro.
// See: https://docs.microsoft.com/en-us/windows-hardware/drivers/kernel/using-ntstatus-values
pub fn nt_success(status: NTSTATUS) -> bool {
    status >= 0
}

impl OBJECT_ATTRIBUTES {
    pub fn with_length() -> Self {
        Self {
            Length: size_of::<Self>() as _,
            RootDirectory: ptr::null_mut(),
            ObjectName: ptr::null_mut(),
            Attributes: 0,
            SecurityDescriptor: ptr::null_mut(),
            SecurityQualityOfService: ptr::null_mut(),
        }
    }
}

impl IO_STATUS_BLOCK {
    pub const PENDING: Self =
        IO_STATUS_BLOCK { Anonymous: IO_STATUS_BLOCK_0 { Status: STATUS_PENDING }, Information: 0 };
    pub fn status(&self) -> NTSTATUS {
        // SAFETY: If `self.Anonymous.Status` was set then this is obviously safe.
        // If `self.Anonymous.Pointer` was set then this is the equivalent to converting
        // the pointer to an integer, which is also safe.
        // Currently the only safe way to construct `IO_STATUS_BLOCK` outside of
        // this module is to call the `default` method, which sets the `Status`.
        unsafe { self.Anonymous.Status }
    }
}

/// NB: Use carefully! In general using this as a reference is likely to get the
/// provenance wrong for the `rest` field!
#[repr(C)]
pub struct REPARSE_DATA_BUFFER {
    pub ReparseTag: c_uint,
    pub ReparseDataLength: c_ushort,
    pub Reserved: c_ushort,
    pub rest: (),
}

/// NB: Use carefully! In general using this as a reference is likely to get the
/// provenance wrong for the `PathBuffer` field!
#[repr(C)]
pub struct SYMBOLIC_LINK_REPARSE_BUFFER {
    pub SubstituteNameOffset: c_ushort,
    pub SubstituteNameLength: c_ushort,
    pub PrintNameOffset: c_ushort,
    pub PrintNameLength: c_ushort,
    pub Flags: c_ulong,
    pub PathBuffer: WCHAR,
}

#[repr(C)]
pub struct MOUNT_POINT_REPARSE_BUFFER {
    pub SubstituteNameOffset: c_ushort,
    pub SubstituteNameLength: c_ushort,
    pub PrintNameOffset: c_ushort,
    pub PrintNameLength: c_ushort,
    pub PathBuffer: WCHAR,
}

// Desktop specific functions & types
#[cfg(not(target_vendor = "uwp"))]
pub const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

// Use raw-dylib to import ProcessPrng as we can't rely on there being an import library.
#[cfg(not(target_vendor = "win7"))]
#[cfg_attr(
    target_arch = "x86",
    link(name = "bcryptprimitives", kind = "raw-dylib", import_name_type = "undecorated")
)]
#[cfg_attr(not(target_arch = "x86"), link(name = "bcryptprimitives", kind = "raw-dylib"))]
unsafe extern "system" {
    pub fn ProcessPrng(pbdata: *mut u8, cbdata: usize) -> BOOL;
}

windows_link::link!("ntdll.dll" "system" fn NtCreateNamedPipeFile(
    filehandle: *mut HANDLE,
    desiredaccess: u32,
    objectattributes: *const OBJECT_ATTRIBUTES,
    iostatusblock: *mut IO_STATUS_BLOCK,
    shareaccess: u32,
    createdisposition: u32,
    createoptions: u32,
    namedpipetype: u32,
    readmode: u32,
    completionmode: u32,
    maximuminstances: u32,
    inboundquota: u32,
    outboundquota: u32,
    defaulttimeout: *const u64,
) -> NTSTATUS);

// Functions that aren't available on every version of Windows that we support,
// but we still use them and just provide some form of a fallback implementation.
compat_fn_with_fallback! {
    pub static KERNEL32: &CStr = c"kernel32";

    // >= Win10 1607
    // https://docs.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setthreaddescription
    pub fn SetThreadDescription(hthread: HANDLE, lpthreaddescription: PCWSTR) -> HRESULT {
        unsafe { SetLastError(ERROR_CALL_NOT_IMPLEMENTED as u32); E_NOTIMPL }
    }

    // >= Win10 1607
    // https://docs.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getthreaddescription
    pub fn GetThreadDescription(hthread: HANDLE, lpthreaddescription: *mut PWSTR) -> HRESULT {
        unsafe { SetLastError(ERROR_CALL_NOT_IMPLEMENTED as u32); E_NOTIMPL }
    }

    // >= Win8 / Server 2012
    // https://docs.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-getsystemtimepreciseasfiletime
    #[cfg(target_vendor = "win7")]
    pub fn GetSystemTimePreciseAsFileTime(lpsystemtimeasfiletime: *mut FILETIME) -> () {
        unsafe { GetSystemTimeAsFileTime(lpsystemtimeasfiletime) }
    }

    // >= Win11 / Server 2022
    // https://docs.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-gettemppath2a
    pub fn GetTempPath2W(bufferlength: u32, buffer: PWSTR) -> u32 {
        unsafe {  GetTempPathW(bufferlength, buffer) }
    }
}

#[cfg(not(target_vendor = "win7"))]
// Use raw-dylib to import synchronization functions to workaround issues with the older mingw import library.
#[cfg_attr(
    target_arch = "x86",
    link(
        name = "api-ms-win-core-synch-l1-2-0",
        kind = "raw-dylib",
        import_name_type = "undecorated"
    )
)]
#[cfg_attr(
    not(target_arch = "x86"),
    link(name = "api-ms-win-core-synch-l1-2-0", kind = "raw-dylib")
)]
unsafe extern "system" {
    pub fn WaitOnAddress(
        address: *const c_void,
        compareaddress: *const c_void,
        addresssize: usize,
        dwmilliseconds: u32,
    ) -> BOOL;
    pub fn WakeByAddressSingle(address: *const c_void);
    pub fn WakeByAddressAll(address: *const c_void);
}

// These are loaded by `load_synch_functions`.
#[cfg(target_vendor = "win7")]
compat_fn_optional! {
    pub fn WaitOnAddress(
        address: *const c_void,
        compareaddress: *const c_void,
        addresssize: usize,
        dwmilliseconds: u32
    ) -> BOOL;
    pub fn WakeByAddressSingle(address: *const c_void);
}

#[cfg(any(target_vendor = "win7"))]
compat_fn_with_fallback! {
    pub static NTDLL: &CStr = c"ntdll";

    #[cfg(target_vendor = "win7")]
    pub fn NtCreateKeyedEvent(
        KeyedEventHandle: *mut HANDLE,
        DesiredAccess: u32,
        ObjectAttributes: *mut c_void,
        Flags: u32
    ) -> NTSTATUS {
        panic!("keyed events not available")
    }
    #[cfg(target_vendor = "win7")]
    pub fn NtReleaseKeyedEvent(
        EventHandle: HANDLE,
        Key: *const c_void,
        Alertable: bool,
        Timeout: *mut i64
    ) -> NTSTATUS {
        panic!("keyed events not available")
    }
    #[cfg(target_vendor = "win7")]
    pub fn NtWaitForKeyedEvent(
        EventHandle: HANDLE,
        Key: *const c_void,
        Alertable: bool,
        Timeout: *mut i64
    ) -> NTSTATUS {
        panic!("keyed events not available")
    }
}

cfg_select! {
    target_vendor = "uwp" => {
        windows_link::link_raw_dylib!("ntdll.dll" "system" fn NtCreateFile(filehandle : *mut HANDLE, desiredaccess : ACCESS_MASK, objectattributes : *const OBJECT_ATTRIBUTES, iostatusblock : *mut IO_STATUS_BLOCK, allocationsize : *const i64, fileattributes : u32, shareaccess : u32, createdisposition : u32, createoptions : u32, eabuffer : *const core::ffi::c_void, ealength : u32) -> NTSTATUS);
        windows_link::link_raw_dylib!("ntdll.dll" "system" fn NtOpenFile(filehandle : *mut HANDLE, desiredaccess : ACCESS_MASK, objectattributes : *const OBJECT_ATTRIBUTES, iostatusblock : *mut IO_STATUS_BLOCK, shareaccess : u32, openoptions : u32) -> NTSTATUS);
        windows_link::link_raw_dylib!("ntdll.dll" "system" fn NtReadFile(filehandle : HANDLE, event : HANDLE, apcroutine : PIO_APC_ROUTINE, apccontext : *const core::ffi::c_void, iostatusblock : *mut IO_STATUS_BLOCK, buffer : *mut core::ffi::c_void, length : u32, byteoffset : *const i64, key : *const u32) -> NTSTATUS);
        windows_link::link_raw_dylib!("ntdll.dll" "system" fn NtWriteFile(filehandle : HANDLE, event : HANDLE, apcroutine : PIO_APC_ROUTINE, apccontext : *const core::ffi::c_void, iostatusblock : *mut IO_STATUS_BLOCK, buffer : *const core::ffi::c_void, length : u32, byteoffset : *const i64, key : *const u32) -> NTSTATUS);
        windows_link::link_raw_dylib!("ntdll.dll" "system" fn RtlNtStatusToDosError(status : NTSTATUS) -> u32);
    }
    _ => {}
}

// Only available starting with Windows 8.
#[cfg(not(target_vendor = "win7"))]
windows_link::link!("ws2_32.dll" "system" fn GetHostNameW(name : PWSTR, namelen : i32) -> i32);

unsafe extern "C" {
    pub fn atexit(cb: unsafe extern "C" fn()) -> c_int;
}
