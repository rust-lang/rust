// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! C definitions used by libnative that don't belong in liblibc

#![allow(bad_style, dead_code, overflowing_literals)]

use libc;

pub use self::GET_FILEEX_INFO_LEVELS::*;
pub use self::FILE_INFO_BY_HANDLE_CLASS::*;
pub use libc::consts::os::extra::{
    FILE_ATTRIBUTE_READONLY,
    FILE_ATTRIBUTE_DIRECTORY,
    WSAPROTOCOL_LEN,
};
pub use libc::types::os::arch::extra::{GROUP, GUID, WSAPROTOCOLCHAIN};

pub const WSADESCRIPTION_LEN: usize = 256;
pub const WSASYS_STATUS_LEN: usize = 128;
pub const FIONBIO: libc::c_long = 0x8004667e;
pub const FD_SETSIZE: usize = 64;
pub const MSG_DONTWAIT: libc::c_int = 0;
pub const ERROR_ILLEGAL_CHARACTER: libc::c_int = 582;
pub const ENABLE_ECHO_INPUT: libc::DWORD = 0x4;
pub const ENABLE_EXTENDED_FLAGS: libc::DWORD = 0x80;
pub const ENABLE_INSERT_MODE: libc::DWORD = 0x20;
pub const ENABLE_LINE_INPUT: libc::DWORD = 0x2;
pub const ENABLE_PROCESSED_INPUT: libc::DWORD = 0x1;
pub const ENABLE_QUICK_EDIT_MODE: libc::DWORD = 0x40;
pub const WSA_INVALID_EVENT: WSAEVENT = 0 as WSAEVENT;

pub const FD_ACCEPT: libc::c_long = 0x08;
pub const FD_MAX_EVENTS: usize = 10;
pub const WSA_INFINITE: libc::DWORD = libc::INFINITE;
pub const WSA_WAIT_TIMEOUT: libc::DWORD = libc::consts::os::extra::WAIT_TIMEOUT;
pub const WSA_WAIT_EVENT_0: libc::DWORD = libc::consts::os::extra::WAIT_OBJECT_0;
pub const WSA_WAIT_FAILED: libc::DWORD = libc::consts::os::extra::WAIT_FAILED;
pub const WSAESHUTDOWN: libc::c_int = 10058;

pub const ERROR_NO_MORE_FILES: libc::DWORD = 18;
pub const TOKEN_READ: libc::DWORD = 0x20008;

#[repr(C)]
#[cfg(target_arch = "x86")]
pub struct WSADATA {
    pub wVersion: libc::WORD,
    pub wHighVersion: libc::WORD,
    pub szDescription: [u8; WSADESCRIPTION_LEN + 1],
    pub szSystemStatus: [u8; WSASYS_STATUS_LEN + 1],
    pub iMaxSockets: u16,
    pub iMaxUdpDg: u16,
    pub lpVendorInfo: *mut u8,
}
#[repr(C)]
#[cfg(target_arch = "x86_64")]
pub struct WSADATA {
    pub wVersion: libc::WORD,
    pub wHighVersion: libc::WORD,
    pub iMaxSockets: u16,
    pub iMaxUdpDg: u16,
    pub lpVendorInfo: *mut u8,
    pub szDescription: [u8; WSADESCRIPTION_LEN + 1],
    pub szSystemStatus: [u8; WSASYS_STATUS_LEN + 1],
}

pub type LPWSADATA = *mut WSADATA;

#[repr(C)]
pub struct WSANETWORKEVENTS {
    pub lNetworkEvents: libc::c_long,
    pub iErrorCode: [libc::c_int; FD_MAX_EVENTS],
}

pub type LPWSANETWORKEVENTS = *mut WSANETWORKEVENTS;

pub type WSAEVENT = libc::HANDLE;

#[repr(C)]
#[derive(Copy)]
pub struct WSAPROTOCOL_INFO {
    pub dwServiceFlags1: libc::DWORD,
    pub dwServiceFlags2: libc::DWORD,
    pub dwServiceFlags3: libc::DWORD,
    pub dwServiceFlags4: libc::DWORD,
    pub dwProviderFlags: libc::DWORD,
    pub ProviderId: GUID,
    pub dwCatalogEntryId: libc::DWORD,
    pub ProtocolChain: WSAPROTOCOLCHAIN,
    pub iVersion: libc::c_int,
    pub iAddressFamily: libc::c_int,
    pub iMaxSockAddr: libc::c_int,
    pub iMinSockAddr: libc::c_int,
    pub iSocketType: libc::c_int,
    pub iProtocol: libc::c_int,
    pub iProtocolMaxOffset: libc::c_int,
    pub iNetworkByteOrder: libc::c_int,
    pub iSecurityScheme: libc::c_int,
    pub dwMessageSize: libc::DWORD,
    pub dwProviderReserved: libc::DWORD,
    pub szProtocol: [u16; (WSAPROTOCOL_LEN as usize) + 1],
}

pub type LPWSAPROTOCOL_INFO = *mut WSAPROTOCOL_INFO;

#[repr(C)]
pub struct fd_set {
    fd_count: libc::c_uint,
    fd_array: [libc::SOCKET; FD_SETSIZE],
}

pub fn fd_set(set: &mut fd_set, s: libc::SOCKET) {
    set.fd_array[set.fd_count as usize] = s;
    set.fd_count += 1;
}

pub type SHORT = libc::c_short;

#[repr(C)]
pub struct COORD {
    pub X: SHORT,
    pub Y: SHORT,
}

#[repr(C)]
pub struct SMALL_RECT {
    pub Left: SHORT,
    pub Top: SHORT,
    pub Right: SHORT,
    pub Bottom: SHORT,
}

#[repr(C)]
pub struct CONSOLE_SCREEN_BUFFER_INFO {
    pub dwSize: COORD,
    pub dwCursorPosition: COORD,
    pub wAttributes: libc::WORD,
    pub srWindow: SMALL_RECT,
    pub dwMaximumWindowSize: COORD,
}
pub type PCONSOLE_SCREEN_BUFFER_INFO = *mut CONSOLE_SCREEN_BUFFER_INFO;

#[repr(C)]
pub struct WIN32_FILE_ATTRIBUTE_DATA {
    pub dwFileAttributes: libc::DWORD,
    pub ftCreationTime: libc::FILETIME,
    pub ftLastAccessTime: libc::FILETIME,
    pub ftLastWriteTime: libc::FILETIME,
    pub nFileSizeHigh: libc::DWORD,
    pub nFileSizeLow: libc::DWORD,
}

#[repr(C)]
pub struct BY_HANDLE_FILE_INFORMATION {
    pub dwFileAttributes: libc::DWORD,
    pub ftCreationTime: libc::FILETIME,
    pub ftLastAccessTime: libc::FILETIME,
    pub ftLastWriteTime: libc::FILETIME,
    pub dwVolumeSerialNumber: libc::DWORD,
    pub nFileSizeHigh: libc::DWORD,
    pub nFileSizeLow: libc::DWORD,
    pub nNumberOfLinks: libc::DWORD,
    pub nFileIndexHigh: libc::DWORD,
    pub nFileIndexLow: libc::DWORD,
}

pub type LPBY_HANDLE_FILE_INFORMATION = *mut BY_HANDLE_FILE_INFORMATION;

#[repr(C)]
pub enum GET_FILEEX_INFO_LEVELS {
    GetFileExInfoStandard,
    GetFileExMaxInfoLevel
}

#[repr(C)]
pub enum FILE_INFO_BY_HANDLE_CLASS {
    FileBasicInfo                   = 0,
    FileStandardInfo                = 1,
    FileNameInfo                    = 2,
    FileRenameInfo                  = 3,
    FileDispositionInfo             = 4,
    FileAllocationInfo              = 5,
    FileEndOfFileInfo               = 6,
    FileStreamInfo                  = 7,
    FileCompressionInfo             = 8,
    FileAttributeTagInfo            = 9,
    FileIdBothDirectoryInfo         = 10, // 0xA
    FileIdBothDirectoryRestartInfo  = 11, // 0xB
    FileIoPriorityHintInfo          = 12, // 0xC
    FileRemoteProtocolInfo          = 13, // 0xD
    FileFullDirectoryInfo           = 14, // 0xE
    FileFullDirectoryRestartInfo    = 15, // 0xF
    FileStorageInfo                 = 16, // 0x10
    FileAlignmentInfo               = 17, // 0x11
    FileIdInfo                      = 18, // 0x12
    FileIdExtdDirectoryInfo         = 19, // 0x13
    FileIdExtdDirectoryRestartInfo  = 20, // 0x14
    MaximumFileInfoByHandlesClass
}

#[repr(C)]
pub struct FILE_END_OF_FILE_INFO {
    pub EndOfFile: libc::LARGE_INTEGER,
}

#[link(name = "ws2_32")]
extern "system" {
    pub fn WSAStartup(wVersionRequested: libc::WORD,
                      lpWSAData: LPWSADATA) -> libc::c_int;
    pub fn WSACleanup() -> libc::c_int;
    pub fn WSAGetLastError() -> libc::c_int;
    pub fn WSACloseEvent(hEvent: WSAEVENT) -> libc::BOOL;
    pub fn WSACreateEvent() -> WSAEVENT;
    pub fn WSAEventSelect(s: libc::SOCKET,
                          hEventObject: WSAEVENT,
                          lNetworkEvents: libc::c_long) -> libc::c_int;
    pub fn WSASetEvent(hEvent: WSAEVENT) -> libc::BOOL;
    pub fn WSAWaitForMultipleEvents(cEvents: libc::DWORD,
                                    lphEvents: *const WSAEVENT,
                                    fWaitAll: libc::BOOL,
                                    dwTimeout: libc::DWORD,
                                    fAltertable: libc::BOOL) -> libc::DWORD;
    pub fn WSAEnumNetworkEvents(s: libc::SOCKET,
                                hEventObject: WSAEVENT,
                                lpNetworkEvents: LPWSANETWORKEVENTS)
                                -> libc::c_int;
    pub fn WSADuplicateSocketW(s: libc::SOCKET,
                               dwProcessId: libc::DWORD,
                               lpProtocolInfo: LPWSAPROTOCOL_INFO)
                               -> libc::c_int;
    pub fn GetCurrentProcessId() -> libc::DWORD;
    pub fn WSASocketW(af: libc::c_int,
                      kind: libc::c_int,
                      protocol: libc::c_int,
                      lpProtocolInfo: LPWSAPROTOCOL_INFO,
                      g: GROUP,
                      dwFlags: libc::DWORD) -> libc::SOCKET;

    pub fn ioctlsocket(s: libc::SOCKET, cmd: libc::c_long,
                       argp: *mut libc::c_ulong) -> libc::c_int;
    pub fn select(nfds: libc::c_int,
                  readfds: *mut fd_set,
                  writefds: *mut fd_set,
                  exceptfds: *mut fd_set,
                  timeout: *mut libc::timeval) -> libc::c_int;
    pub fn getsockopt(sockfd: libc::SOCKET,
                      level: libc::c_int,
                      optname: libc::c_int,
                      optval: *mut libc::c_char,
                      optlen: *mut libc::c_int) -> libc::c_int;

    pub fn SetEvent(hEvent: libc::HANDLE) -> libc::BOOL;
    pub fn WaitForMultipleObjects(nCount: libc::DWORD,
                                  lpHandles: *const libc::HANDLE,
                                  bWaitAll: libc::BOOL,
                                  dwMilliseconds: libc::DWORD) -> libc::DWORD;

    pub fn CancelIo(hFile: libc::HANDLE) -> libc::BOOL;
    pub fn CancelIoEx(hFile: libc::HANDLE,
                      lpOverlapped: libc::LPOVERLAPPED) -> libc::BOOL;
}

pub mod compat {
    use prelude::v1::*;

    use ffi::CString;
    use libc::types::os::arch::extra::{LPCWSTR, HMODULE, LPCSTR, LPVOID};
    use sync::atomic::{AtomicUsize, Ordering};

    extern "system" {
        fn GetModuleHandleW(lpModuleName: LPCWSTR) -> HMODULE;
        fn GetProcAddress(hModule: HMODULE, lpProcName: LPCSTR) -> LPVOID;
    }

    fn store_func(ptr: &AtomicUsize, module: &str, symbol: &str,
                  fallback: usize) -> usize {
        let mut module: Vec<u16> = module.utf16_units().collect();
        module.push(0);
        let symbol = CString::new(symbol).unwrap();
        let func = unsafe {
            let handle = GetModuleHandleW(module.as_ptr());
            GetProcAddress(handle, symbol.as_ptr()) as usize
        };
        let value = if func == 0 {fallback} else {func};
        ptr.store(value, Ordering::SeqCst);
        value
    }

    /// Macro for creating a compatibility fallback for a Windows function
    ///
    /// # Example
    /// ```
    /// compat_fn!(adll32::SomeFunctionW(_arg: LPCWSTR) {
    ///     // Fallback implementation
    /// })
    /// ```
    ///
    /// Note that arguments unused by the fallback implementation should not be
    /// called `_` as they are used to be passed to the real function if
    /// available.
    macro_rules! compat_fn {
        ($module:ident::$symbol:ident($($argname:ident: $argtype:ty),*)
                                      -> $rettype:ty { $fallback:expr }) => (
            #[inline(always)]
            pub unsafe fn $symbol($($argname: $argtype),*) -> $rettype {
                use sync::atomic::{AtomicUsize, ATOMIC_USIZE_INIT, Ordering};
                use mem;

                static PTR: AtomicUsize = ATOMIC_USIZE_INIT;

                fn load() -> usize {
                    ::sys::c::compat::store_func(&PTR,
                                                 stringify!($module),
                                                 stringify!($symbol),
                                                 fallback as usize)
                }

                extern "system" fn fallback($($argname: $argtype),*)
                                            -> $rettype { $fallback }

                let addr = match PTR.load(Ordering::SeqCst) {
                    0 => load(),
                    n => n,
                };
                let f: extern "system" fn($($argtype),*) -> $rettype =
                    mem::transmute(addr);
                f($($argname),*)
            }
        )
    }

    /// Compatibility layer for functions in `kernel32.dll`
    ///
    /// Latest versions of Windows this is needed for:
    ///
    /// * `CreateSymbolicLinkW`: Windows XP, Windows Server 2003
    /// * `GetFinalPathNameByHandleW`: Windows XP, Windows Server 2003
    pub mod kernel32 {
        use libc::c_uint;
        use libc::types::os::arch::extra::{DWORD, LPCWSTR, BOOLEAN, HANDLE};
        use libc::consts::os::extra::ERROR_CALL_NOT_IMPLEMENTED;
        use sys::c::SetLastError;

        compat_fn! {
            kernel32::CreateSymbolicLinkW(_lpSymlinkFileName: LPCWSTR,
                                          _lpTargetFileName: LPCWSTR,
                                          _dwFlags: DWORD) -> BOOLEAN {
                unsafe { SetLastError(ERROR_CALL_NOT_IMPLEMENTED as DWORD); 0 }
            }
        }

        compat_fn! {
            kernel32::GetFinalPathNameByHandleW(_hFile: HANDLE,
                                                _lpszFilePath: LPCWSTR,
                                                _cchFilePath: DWORD,
                                                _dwFlags: DWORD) -> DWORD {
                unsafe { SetLastError(ERROR_CALL_NOT_IMPLEMENTED as DWORD); 0 }
            }
        }

        compat_fn! {
            kernel32::SetThreadErrorMode(_dwNewMode: DWORD, _lpOldMode: *mut DWORD) -> c_uint {
                unsafe { SetLastError(ERROR_CALL_NOT_IMPLEMENTED as DWORD); 0 }
            }
        }
    }
}

extern "system" {
    // FIXME - pInputControl should be PCONSOLE_READCONSOLE_CONTROL
    pub fn ReadConsoleW(hConsoleInput: libc::HANDLE,
                        lpBuffer: libc::LPVOID,
                        nNumberOfCharsToRead: libc::DWORD,
                        lpNumberOfCharsRead: libc::LPDWORD,
                        pInputControl: libc::LPVOID) -> libc::BOOL;

    pub fn WriteConsoleW(hConsoleOutput: libc::HANDLE,
                         lpBuffer: libc::types::os::arch::extra::LPCVOID,
                         nNumberOfCharsToWrite: libc::DWORD,
                         lpNumberOfCharsWritten: libc::LPDWORD,
                         lpReserved: libc::LPVOID) -> libc::BOOL;

    pub fn GetConsoleMode(hConsoleHandle: libc::HANDLE,
                          lpMode: libc::LPDWORD) -> libc::BOOL;

    pub fn SetConsoleMode(hConsoleHandle: libc::HANDLE,
                          lpMode: libc::DWORD) -> libc::BOOL;
    pub fn GetConsoleScreenBufferInfo(
        hConsoleOutput: libc::HANDLE,
        lpConsoleScreenBufferInfo: PCONSOLE_SCREEN_BUFFER_INFO,
    ) -> libc::BOOL;

    pub fn GetFileAttributesExW(lpFileName: libc::LPCWSTR,
                                fInfoLevelId: GET_FILEEX_INFO_LEVELS,
                                lpFileInformation: libc::LPVOID) -> libc::BOOL;
    pub fn RemoveDirectoryW(lpPathName: libc::LPCWSTR) -> libc::BOOL;
    pub fn SetFileAttributesW(lpFileName: libc::LPCWSTR,
                              dwFileAttributes: libc::DWORD) -> libc::BOOL;
    pub fn GetFileAttributesW(lpFileName: libc::LPCWSTR) -> libc::DWORD;
    pub fn GetFileInformationByHandle(hFile: libc::HANDLE,
                            lpFileInformation: LPBY_HANDLE_FILE_INFORMATION)
                            -> libc::BOOL;

    pub fn SetLastError(dwErrCode: libc::DWORD);
    pub fn GetCommandLineW() -> *mut libc::LPCWSTR;
    pub fn LocalFree(ptr: *mut libc::c_void);
    pub fn CommandLineToArgvW(lpCmdLine: *mut libc::LPCWSTR,
                              pNumArgs: *mut libc::c_int) -> *mut *mut u16;
    pub fn SetFileTime(hFile: libc::HANDLE,
                       lpCreationTime: *const libc::FILETIME,
                       lpLastAccessTime: *const libc::FILETIME,
                       lpLastWriteTime: *const libc::FILETIME) -> libc::BOOL;
    pub fn SetFileInformationByHandle(hFile: libc::HANDLE,
                    FileInformationClass: FILE_INFO_BY_HANDLE_CLASS,
                    lpFileInformation: libc::LPVOID,
                    dwBufferSize: libc::DWORD) -> libc::BOOL;
    pub fn GetTempPathW(nBufferLength: libc::DWORD,
                        lpBuffer: libc::LPCWSTR) -> libc::DWORD;
    pub fn OpenProcessToken(ProcessHandle: libc::HANDLE,
                            DesiredAccess: libc::DWORD,
                            TokenHandle: *mut libc::HANDLE) -> libc::BOOL;
    pub fn GetCurrentProcess() -> libc::HANDLE;
}

#[link(name = "userenv")]
extern "system" {
    pub fn GetUserProfileDirectoryW(hToken: libc::HANDLE,
                                    lpProfileDir: libc::LPCWSTR,
                                    lpcchSize: *mut libc::DWORD) -> libc::BOOL;
}
