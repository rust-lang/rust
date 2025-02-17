//! C definitions used by libnative that don't belong in liblibc

#![allow(nonstandard_style)]
#![cfg_attr(test, allow(dead_code))]
#![unstable(issue = "none", feature = "windows_c")]
#![allow(clippy::style)]

use core::ffi::{CStr, c_uint, c_ulong, c_ushort, c_void};
use core::{mem, ptr};

mod windows_sys;
pub use windows_sys::*;

pub type WCHAR = u16;

pub const INVALID_HANDLE_VALUE: HANDLE = ::core::ptr::without_provenance_mut(-1i32 as _);

// https://learn.microsoft.com/en-us/cpp/c-runtime-library/exit-success-exit-failure?view=msvc-170
pub const EXIT_SUCCESS: u32 = 0;
pub const EXIT_FAILURE: u32 = 1;

#[cfg(target_vendor = "win7")]
pub const CONDITION_VARIABLE_INIT: CONDITION_VARIABLE = CONDITION_VARIABLE { Ptr: ptr::null_mut() };
#[cfg(target_vendor = "win7")]
pub const SRWLOCK_INIT: SRWLOCK = SRWLOCK { Ptr: ptr::null_mut() };
#[cfg(not(target_thread_local))]
pub const INIT_ONCE_STATIC_INIT: INIT_ONCE = INIT_ONCE { Ptr: ptr::null_mut() };

// Some windows_sys types have different signs than the types we use.
pub const OBJ_DONT_REPARSE: u32 = windows_sys::OBJ_DONT_REPARSE as u32;
pub const FRS_ERR_SYSVOL_POPULATE_TIMEOUT: u32 =
    windows_sys::FRS_ERR_SYSVOL_POPULATE_TIMEOUT as u32;

// Equivalent to the `NT_SUCCESS` C preprocessor macro.
// See: https://docs.microsoft.com/en-us/windows-hardware/drivers/kernel/using-ntstatus-values
pub fn nt_success(status: NTSTATUS) -> bool {
    status >= 0
}

impl UNICODE_STRING {
    pub fn from_ref(slice: &[u16]) -> Self {
        let len = mem::size_of_val(slice);
        Self { Length: len as _, MaximumLength: len as _, Buffer: slice.as_ptr() as _ }
    }
}

impl Default for OBJECT_ATTRIBUTES {
    fn default() -> Self {
        Self {
            Length: mem::size_of::<Self>() as _,
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
cfg_if::cfg_if! {
if #[cfg(not(target_vendor = "uwp"))] {
    pub const EXCEPTION_CONTINUE_SEARCH: i32 = 0;
}
}

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

#[cfg(any(target_vendor = "win7", target_vendor = "uwp"))]
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
        Alertable: BOOLEAN,
        Timeout: *mut i64
    ) -> NTSTATUS {
        panic!("keyed events not available")
    }
    #[cfg(target_vendor = "win7")]
    pub fn NtWaitForKeyedEvent(
        EventHandle: HANDLE,
        Key: *const c_void,
        Alertable: BOOLEAN,
        Timeout: *mut i64
    ) -> NTSTATUS {
        panic!("keyed events not available")
    }

    // These functions are available on UWP when lazily loaded. They will fail WACK if loaded statically.
    #[cfg(target_vendor = "uwp")]
    pub fn NtCreateFile(
        filehandle: *mut HANDLE,
        desiredaccess: FILE_ACCESS_RIGHTS,
        objectattributes: *const OBJECT_ATTRIBUTES,
        iostatusblock: *mut IO_STATUS_BLOCK,
        allocationsize: *const i64,
        fileattributes: FILE_FLAGS_AND_ATTRIBUTES,
        shareaccess: FILE_SHARE_MODE,
        createdisposition: NTCREATEFILE_CREATE_DISPOSITION,
        createoptions: NTCREATEFILE_CREATE_OPTIONS,
        eabuffer: *const c_void,
        ealength: u32
    ) -> NTSTATUS {
        STATUS_NOT_IMPLEMENTED
    }
    #[cfg(target_vendor = "uwp")]
    pub fn NtReadFile(
        filehandle: HANDLE,
        event: HANDLE,
        apcroutine: PIO_APC_ROUTINE,
        apccontext: *const c_void,
        iostatusblock: *mut IO_STATUS_BLOCK,
        buffer: *mut c_void,
        length: u32,
        byteoffset: *const i64,
        key: *const u32
    ) -> NTSTATUS {
        STATUS_NOT_IMPLEMENTED
    }
    #[cfg(target_vendor = "uwp")]
    pub fn NtWriteFile(
        filehandle: HANDLE,
        event: HANDLE,
        apcroutine: PIO_APC_ROUTINE,
        apccontext: *const c_void,
        iostatusblock: *mut IO_STATUS_BLOCK,
        buffer: *const c_void,
        length: u32,
        byteoffset: *const i64,
        key: *const u32
    ) -> NTSTATUS {
        STATUS_NOT_IMPLEMENTED
    }
    #[cfg(target_vendor = "uwp")]
    pub fn RtlNtStatusToDosError(Status: NTSTATUS) -> u32 {
        Status as u32
    }
}
