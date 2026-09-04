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

// Short-hand for cast_unsigned.
// It prevents most of the casts below from spilling over multiple lines.
const fn cu(n: i32) -> u32 {
    n.cast_unsigned()
}

// dwShareMode -- CreateFile
pub const FILE_SHARE_DELETE: u32 = cu(windows_sys::FILE_SHARE_DELETE);
pub const FILE_SHARE_READ: u32 = cu(windows_sys::FILE_SHARE_READ);
pub const FILE_SHARE_WRITE: u32 = cu(windows_sys::FILE_SHARE_WRITE);

// CreateOptions (ULONG) -- NtCreateFile
pub const FILE_DIRECTORY_FILE: u32 = cu(windows_sys::FILE_DIRECTORY_FILE);
pub const FILE_SYNCHRONOUS_IO_NONALERT: u32 = cu(windows_sys::FILE_SYNCHRONOUS_IO_NONALERT);
pub const FILE_NON_DIRECTORY_FILE: u32 = cu(windows_sys::FILE_NON_DIRECTORY_FILE);
pub const FILE_OPEN_REPARSE_POINT: u32 = cu(windows_sys::FILE_OPEN_REPARSE_POINT);

// dwDesiredAccess -- many, many functions
pub const DELETE: u32 = cu(windows_sys::DELETE);
pub const SYNCHRONIZE: u32 = cu(windows_sys::SYNCHRONIZE);
pub const GENERIC_WRITE: u32 = cu(windows_sys::GENERIC_WRITE);
pub const FILE_GENERIC_WRITE: u32 = cu(windows_sys::FILE_GENERIC_WRITE);
pub const FILE_LIST_DIRECTORY: u32 = cu(windows_sys::FILE_LIST_DIRECTORY);
pub const FILE_WRITE_DATA: u32 = cu(windows_sys::FILE_WRITE_DATA);
pub const FILE_READ_ATTRIBUTES: u32 = cu(windows_sys::FILE_READ_ATTRIBUTES);
pub const FILE_WRITE_ATTRIBUTES: u32 = cu(windows_sys::FILE_WRITE_ATTRIBUTES);
pub const FILE_TRAVERSE: u32 = cu(windows_sys::FILE_TRAVERSE);
pub const TIMER_ALL_ACCESS: u32 = cu(windows_sys::TIMER_ALL_ACCESS);
#[cfg(target_vendor = "win7")]
pub const TOKEN_READ: u32 = cu(windows_sys::TOKEN_READ);

// dwOptions -- DuplicateHandle
pub const DUPLICATE_SAME_ACCESS: u32 = cu(windows_sys::DUPLICATE_SAME_ACCESS);

// dwFlags -- GetModuleHandleEx
pub const GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS: u32 =
    cu(windows_sys::GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS);
pub const GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT: u32 =
    cu(windows_sys::GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT);

// dwSecurityQosFlags -- CreateFile
pub const SECURITY_SQOS_PRESENT: u32 = cu(windows_sys::SECURITY_SQOS_PRESENT);

// dwFlags -- LockFileEx
pub const LOCKFILE_EXCLUSIVE_LOCK: u32 = cu(windows_sys::LOCKFILE_EXCLUSIVE_LOCK);
pub const LOCKFILE_FAIL_IMMEDIATELY: u32 = cu(windows_sys::LOCKFILE_FAIL_IMMEDIATELY);

// dwFileFlags -- CreateFile
pub const FILE_FLAG_BACKUP_SEMANTICS: u32 = cu(windows_sys::FILE_FLAG_BACKUP_SEMANTICS);
pub const FILE_FLAG_OPEN_REPARSE_POINT: u32 = cu(windows_sys::FILE_FLAG_OPEN_REPARSE_POINT);
pub const FILE_FLAG_POSIX_SEMANTICS: u32 = cu(windows_sys::FILE_FLAG_POSIX_SEMANTICS);

// dwFileAttributes -- CreateFile
pub const FILE_ATTRIBUTE_DIRECTORY: u32 = cu(windows_sys::FILE_ATTRIBUTE_DIRECTORY);
pub const FILE_ATTRIBUTE_NORMAL: u32 = cu(windows_sys::FILE_ATTRIBUTE_NORMAL);
pub const FILE_ATTRIBUTE_READONLY: u32 = cu(windows_sys::FILE_ATTRIBUTE_READONLY);
pub const FILE_ATTRIBUTE_REPARSE_POINT: u32 = cu(windows_sys::FILE_ATTRIBUTE_REPARSE_POINT);

// dwFlags -- WSASocket
pub const WSA_FLAG_NO_HANDLE_INHERIT: u32 = cu(windows_sys::WSA_FLAG_NO_HANDLE_INHERIT);
pub const WSA_FLAG_OVERLAPPED: u32 = cu(windows_sys::WSA_FLAG_OVERLAPPED);

// Flags (ULONG) -- FILE_RENAME_INFORMATION
pub const FILE_RENAME_FLAG_REPLACE_IF_EXISTS: u32 =
    cu(windows_sys::FILE_RENAME_FLAG_REPLACE_IF_EXISTS);
pub const FILE_RENAME_FLAG_POSIX_SEMANTICS: u32 = cu(windows_sys::FILE_RENAME_FLAG_POSIX_SEMANTICS);

// Attributes (ULONG) -- OBJECT_ATTRIBUTES
pub const OBJ_INHERIT: u32 = cu(windows_sys::OBJ_INHERIT);
pub const OBJ_DONT_REPARSE: u32 = cu(windows_sys::OBJ_DONT_REPARSE);

// Flags (DWORD) -- FILE_DISPOSITION_INFO_EX
pub const FILE_DISPOSITION_FLAG_DELETE: u32 = cu(windows_sys::FILE_DISPOSITION_FLAG_DELETE);
pub const FILE_DISPOSITION_FLAG_POSIX_SEMANTICS: u32 =
    cu(windows_sys::FILE_DISPOSITION_FLAG_POSIX_SEMANTICS);
pub const FILE_DISPOSITION_FLAG_IGNORE_READONLY_ATTRIBUTE: u32 =
    cu(windows_sys::FILE_DISPOSITION_FLAG_IGNORE_READONLY_ATTRIBUTE);

// Flags (ULONG) -- SymbolicLinkReparseBuffer
pub const SYMLINK_FLAG_RELATIVE: u32 = cu(windows_sys::SYMLINK_FLAG_RELATIVE);

// dwFlags -- CreateSymbolicLink
pub const SYMBOLIC_LINK_FLAG_DIRECTORY: u32 = cu(windows_sys::SYMBOLIC_LINK_FLAG_DIRECTORY);
pub const SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE: u32 =
    cu(windows_sys::SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE);

// dwCreationFlag -- CreateProcess
pub const DETACHED_PROCESS: u32 = cu(windows_sys::DETACHED_PROCESS);
pub const CREATE_NEW_PROCESS_GROUP: u32 = cu(windows_sys::CREATE_NEW_PROCESS_GROUP);
pub const CREATE_UNICODE_ENVIRONMENT: u32 = cu(windows_sys::CREATE_UNICODE_ENVIRONMENT);
pub const EXTENDED_STARTUPINFO_PRESENT: u32 = cu(windows_sys::EXTENDED_STARTUPINFO_PRESENT);

// dwFlags -- STARTUPINFO
pub const STARTF_USESTDHANDLES: u32 = cu(windows_sys::STARTF_USESTDHANDLES);
pub const STARTF_USESHOWWINDOW: u32 = cu(windows_sys::STARTF_USESHOWWINDOW);
pub const STARTF_UNTRUSTEDSOURCE: u32 = cu(windows_sys::STARTF_UNTRUSTEDSOURCE);
pub const STARTF_FORCEONFEEDBACK: u32 = cu(windows_sys::STARTF_FORCEONFEEDBACK);
pub const STARTF_FORCEOFFFEEDBACK: u32 = cu(windows_sys::STARTF_FORCEOFFFEEDBACK);
pub const STARTF_RUNFULLSCREEN: u32 = cu(windows_sys::STARTF_RUNFULLSCREEN);

// dwFlags -- MoveFileEx
pub const MOVEFILE_REPLACE_EXISTING: u32 = cu(windows_sys::MOVEFILE_REPLACE_EXISTING);

// dwFlags -- GetFinalPathNameByHandle
pub const VOLUME_NAME_DOS: u32 = cu(windows_sys::VOLUME_NAME_DOS);
pub const VOLUME_NAME_NT: u32 = cu(windows_sys::VOLUME_NAME_NT);

// dwFlags -- FormatMessageW
pub const FORMAT_MESSAGE_FROM_SYSTEM: u32 = cu(windows_sys::FORMAT_MESSAGE_FROM_SYSTEM);
pub const FORMAT_MESSAGE_IGNORE_INSERTS: u32 = cu(windows_sys::FORMAT_MESSAGE_IGNORE_INSERTS);
pub const FORMAT_MESSAGE_FROM_HMODULE: u32 = cu(windows_sys::FORMAT_MESSAGE_FROM_HMODULE);

// dwFlags -- SetHandleInformation
#[cfg(not(target_vendor = "uwp"))]
pub const HANDLE_FLAG_INHERIT: u32 = cu(windows_sys::HANDLE_FLAG_INHERIT);

// dwCreationFlags -- CreateThread
pub const STACK_SIZE_PARAM_IS_A_RESERVATION: u32 =
    cu(windows_sys::STACK_SIZE_PARAM_IS_A_RESERVATION);

// dwFlags -- WideCharToMultiByte
pub const WC_ERR_INVALID_CHARS: u32 = cu(windows_sys::WC_ERR_INVALID_CHARS);

// dwFlags -- MultiByteToWideChar
pub const MB_ERR_INVALID_CHARS: u32 = cu(windows_sys::MB_ERR_INVALID_CHARS);

// dwFlags -- CreateWaitableTimerExW
pub const CREATE_WAITABLE_TIMER_HIGH_RESOLUTION: u32 =
    cu(windows_sys::CREATE_WAITABLE_TIMER_HIGH_RESOLUTION);

// dwCreationDisposition -- CreateFile
pub const CREATE_NEW: u32 = cu(windows_sys::CREATE_NEW);
pub const OPEN_ALWAYS: u32 = cu(windows_sys::OPEN_ALWAYS);
pub const OPEN_EXISTING: u32 = cu(windows_sys::OPEN_EXISTING);
pub const TRUNCATE_EXISTING: u32 = cu(windows_sys::TRUNCATE_EXISTING);

// CreateDisposition (ULONG) -- NtCreateFile
pub const FILE_OPEN: u32 = cu(windows_sys::FILE_OPEN);
pub const FILE_CREATE: u32 = cu(windows_sys::FILE_CREATE);
pub const FILE_OPEN_IF: u32 = cu(windows_sys::FILE_OPEN_IF);
pub const FILE_OVERWRITE: u32 = cu(windows_sys::FILE_OVERWRITE);
pub const FILE_OVERWRITE_IF: u32 = cu(windows_sys::FILE_OVERWRITE_IF);

// NamedPipeType (ULONG) -- NtCreateNamedPipeFile
pub const FILE_PIPE_BYTE_STREAM_TYPE: u32 = cu(windows_sys::FILE_PIPE_BYTE_STREAM_TYPE);

// ReadMode (ULONG) -- NtCreateNamedPipeFile
pub const FILE_PIPE_BYTE_STREAM_MODE: u32 = cu(windows_sys::FILE_PIPE_BYTE_STREAM_MODE);

// CompletionMode (ULONG) -- NtCreateNamedPipeFile
pub const FILE_PIPE_QUEUE_OPERATION: u32 = cu(windows_sys::FILE_PIPE_QUEUE_OPERATION);

// dwMoveMethod -- SetFilePointerEx
pub const FILE_BEGIN: u32 = cu(windows_sys::FILE_BEGIN);
pub const FILE_END: u32 = cu(windows_sys::FILE_END);
pub const FILE_CURRENT: u32 = cu(windows_sys::FILE_CURRENT);

// dwIoControlCode -- DeviceIoControl
pub const FSCTL_GET_REPARSE_POINT: u32 = cu(windows_sys::FSCTL_GET_REPARSE_POINT);
pub const FSCTL_SET_REPARSE_POINT: u32 = cu(windows_sys::FSCTL_SET_REPARSE_POINT);

// GetFileType return value (DWORD)
pub const FILE_TYPE_PIPE: u32 = cu(windows_sys::FILE_TYPE_PIPE);

// GetACP return value (UINT)
pub const CP_UTF8: u32 = windows_sys::CP_UTF8.cast_unsigned();

// WaitForSingleObject return value (DWORD)
pub const WAIT_OBJECT_0: u32 = windows_sys::WAIT_OBJECT_0.cast_unsigned();

// LPPROGRESS_ROUTINE return value (DWORD)
pub const PROGRESS_CONTINUE: u32 = cu(windows_sys::PROGRESS_CONTINUE);

// Errors
pub const ERROR_SUCCESS: u32 = cu(windows_sys::ERROR_SUCCESS);
pub const ERROR_ACCESS_DENIED: u32 = cu(windows_sys::ERROR_ACCESS_DENIED);
pub const ERROR_ALREADY_EXISTS: u32 = cu(windows_sys::ERROR_ALREADY_EXISTS);
pub const ERROR_BAD_NETPATH: u32 = cu(windows_sys::ERROR_BAD_NETPATH);
pub const ERROR_BAD_NET_NAME: u32 = cu(windows_sys::ERROR_BAD_NET_NAME);
pub const ERROR_CANT_ACCESS_FILE: u32 = cu(windows_sys::ERROR_CANT_ACCESS_FILE);
pub const ERROR_DELETE_PENDING: u32 = cu(windows_sys::ERROR_DELETE_PENDING);
pub const ERROR_DIRECTORY: u32 = cu(windows_sys::ERROR_DIRECTORY);
pub const ERROR_DIR_NOT_EMPTY: u32 = cu(windows_sys::ERROR_DIR_NOT_EMPTY);
pub const ERROR_FILE_NOT_FOUND: u32 = cu(windows_sys::ERROR_FILE_NOT_FOUND);
pub const ERROR_INSUFFICIENT_BUFFER: u32 = cu(windows_sys::ERROR_INSUFFICIENT_BUFFER);
pub const ERROR_INVALID_FUNCTION: u32 = cu(windows_sys::ERROR_INVALID_FUNCTION);
pub const ERROR_INVALID_HANDLE: u32 = cu(windows_sys::ERROR_INVALID_HANDLE);
pub const ERROR_INVALID_PARAMETER: u32 = cu(windows_sys::ERROR_INVALID_PARAMETER);
pub const ERROR_NOT_FOUND: u32 = cu(windows_sys::ERROR_NOT_FOUND);
pub const ERROR_NOT_SUPPORTED: u32 = cu(windows_sys::ERROR_NOT_SUPPORTED);
pub const ERROR_NO_MORE_FILES: u32 = cu(windows_sys::ERROR_NO_MORE_FILES);
pub const ERROR_OPERATION_ABORTED: u32 = cu(windows_sys::ERROR_OPERATION_ABORTED);
pub const ERROR_PATH_NOT_FOUND: u32 = cu(windows_sys::ERROR_PATH_NOT_FOUND);
pub const ERROR_SHARING_VIOLATION: u32 = cu(windows_sys::ERROR_SHARING_VIOLATION);
pub const ERROR_TIMEOUT: u32 = cu(windows_sys::ERROR_TIMEOUT);
pub const ERROR_FILE_EXISTS: u32 = cu(windows_sys::ERROR_FILE_EXISTS);
pub const ERROR_BROKEN_PIPE: u32 = cu(windows_sys::ERROR_BROKEN_PIPE);
pub const ERROR_INVALID_DRIVE: u32 = cu(windows_sys::ERROR_INVALID_DRIVE);
pub const ERROR_NO_DATA: u32 = cu(windows_sys::ERROR_NO_DATA);
pub const ERROR_INVALID_NAME: u32 = cu(windows_sys::ERROR_INVALID_NAME);
pub const ERROR_BAD_PATHNAME: u32 = cu(windows_sys::ERROR_BAD_PATHNAME);
pub const ERROR_NOT_ENOUGH_MEMORY: u32 = cu(windows_sys::ERROR_NOT_ENOUGH_MEMORY);
pub const ERROR_OUTOFMEMORY: u32 = cu(windows_sys::ERROR_OUTOFMEMORY);
pub const ERROR_SEM_TIMEOUT: u32 = cu(windows_sys::ERROR_SEM_TIMEOUT);
pub const ERROR_DRIVER_CANCEL_TIMEOUT: u32 = cu(windows_sys::ERROR_DRIVER_CANCEL_TIMEOUT);
pub const ERROR_SERVICE_REQUEST_TIMEOUT: u32 = cu(windows_sys::ERROR_SERVICE_REQUEST_TIMEOUT);
pub const ERROR_COUNTER_TIMEOUT: u32 = cu(windows_sys::ERROR_COUNTER_TIMEOUT);
pub const ERROR_RESOURCE_CALL_TIMED_OUT: u32 = cu(windows_sys::ERROR_RESOURCE_CALL_TIMED_OUT);
pub const ERROR_CTX_MODEM_RESPONSE_TIMEOUT: u32 = cu(windows_sys::ERROR_CTX_MODEM_RESPONSE_TIMEOUT);
pub const ERROR_CTX_CLIENT_QUERY_TIMEOUT: u32 = cu(windows_sys::ERROR_CTX_CLIENT_QUERY_TIMEOUT);
pub const ERROR_DS_TIMELIMIT_EXCEEDED: u32 = cu(windows_sys::ERROR_DS_TIMELIMIT_EXCEEDED);
pub const DNS_ERROR_RECORD_TIMED_OUT: u32 = cu(windows_sys::DNS_ERROR_RECORD_TIMED_OUT);
pub const ERROR_IPSEC_IKE_TIMED_OUT: u32 = cu(windows_sys::ERROR_IPSEC_IKE_TIMED_OUT);
pub const ERROR_RUNLEVEL_SWITCH_TIMEOUT: u32 = cu(windows_sys::ERROR_RUNLEVEL_SWITCH_TIMEOUT);
pub const ERROR_RUNLEVEL_SWITCH_AGENT_TIMEOUT: u32 =
    cu(windows_sys::ERROR_RUNLEVEL_SWITCH_AGENT_TIMEOUT);
pub const ERROR_CALL_NOT_IMPLEMENTED: u32 = cu(windows_sys::ERROR_CALL_NOT_IMPLEMENTED);
pub const ERROR_HOST_UNREACHABLE: u32 = cu(windows_sys::ERROR_HOST_UNREACHABLE);
pub const ERROR_NETWORK_UNREACHABLE: u32 = cu(windows_sys::ERROR_NETWORK_UNREACHABLE);
pub const ERROR_DIRECTORY_NOT_SUPPORTED: u32 = cu(windows_sys::ERROR_DIRECTORY_NOT_SUPPORTED);
pub const ERROR_WRITE_PROTECT: u32 = cu(windows_sys::ERROR_WRITE_PROTECT);
pub const ERROR_DISK_FULL: u32 = cu(windows_sys::ERROR_DISK_FULL);
pub const ERROR_HANDLE_DISK_FULL: u32 = cu(windows_sys::ERROR_HANDLE_DISK_FULL);
pub const ERROR_SEEK_ON_DEVICE: u32 = cu(windows_sys::ERROR_SEEK_ON_DEVICE);
pub const ERROR_DISK_QUOTA_EXCEEDED: u32 = cu(windows_sys::ERROR_DISK_QUOTA_EXCEEDED);
pub const ERROR_FILE_TOO_LARGE: u32 = cu(windows_sys::ERROR_FILE_TOO_LARGE);
pub const ERROR_BUSY: u32 = cu(windows_sys::ERROR_BUSY);
pub const ERROR_POSSIBLE_DEADLOCK: u32 = cu(windows_sys::ERROR_POSSIBLE_DEADLOCK);
pub const ERROR_NOT_SAME_DEVICE: u32 = cu(windows_sys::ERROR_NOT_SAME_DEVICE);
pub const ERROR_TOO_MANY_LINKS: u32 = cu(windows_sys::ERROR_TOO_MANY_LINKS);
pub const ERROR_TOO_MANY_OPEN_FILES: u32 = cu(windows_sys::ERROR_TOO_MANY_OPEN_FILES);
pub const ERROR_FILENAME_EXCED_RANGE: u32 = cu(windows_sys::ERROR_FILENAME_EXCED_RANGE);
pub const ERROR_CANT_RESOLVE_FILENAME: u32 = cu(windows_sys::ERROR_CANT_RESOLVE_FILENAME);
pub const ERROR_IO_DEVICE: u32 = cu(windows_sys::ERROR_IO_DEVICE);
pub const ERROR_NEGATIVE_SEEK: u32 = cu(windows_sys::ERROR_NEGATIVE_SEEK);
pub const WAIT_TIMEOUT: u32 = cu(windows_sys::WAIT_TIMEOUT);
pub const FRS_ERR_SYSVOL_POPULATE_TIMEOUT: u32 = cu(windows_sys::FRS_ERR_SYSVOL_POPULATE_TIMEOUT);

// ADDRESS_FAMILY
pub const AF_INET: ADDRESS_FAMILY = windows_sys::AF_INET as ADDRESS_FAMILY;
pub const AF_INET6: ADDRESS_FAMILY = windows_sys::AF_INET6 as ADDRESS_FAMILY;
pub const AF_UNIX: ADDRESS_FAMILY = windows_sys::AF_UNIX as ADDRESS_FAMILY;

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
