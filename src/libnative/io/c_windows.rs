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

#![allow(type_overflow)]

use libc;

pub static WSADESCRIPTION_LEN: uint = 256;
pub static WSASYS_STATUS_LEN: uint = 128;
pub static FIONBIO: libc::c_long = 0x8004667e;
static FD_SETSIZE: uint = 64;
pub static MSG_DONTWAIT: libc::c_int = 0;
pub static ERROR_ILLEGAL_CHARACTER: libc::c_int = 582;
pub static ENABLE_ECHO_INPUT: libc::DWORD = 0x4;
pub static ENABLE_EXTENDED_FLAGS: libc::DWORD = 0x80;
pub static ENABLE_INSERT_MODE: libc::DWORD = 0x20;
pub static ENABLE_LINE_INPUT: libc::DWORD = 0x2;
pub static ENABLE_PROCESSED_INPUT: libc::DWORD = 0x1;
pub static ENABLE_QUICK_EDIT_MODE: libc::DWORD = 0x40;
pub static WSA_INVALID_EVENT: WSAEVENT = 0 as WSAEVENT;

pub static FD_ACCEPT: libc::c_long = 0x08;
pub static FD_MAX_EVENTS: uint = 10;
pub static WSA_INFINITE: libc::DWORD = libc::INFINITE;
pub static WSA_WAIT_TIMEOUT: libc::DWORD = libc::consts::os::extra::WAIT_TIMEOUT;
pub static WSA_WAIT_EVENT_0: libc::DWORD = libc::consts::os::extra::WAIT_OBJECT_0;
pub static WSA_WAIT_FAILED: libc::DWORD = libc::consts::os::extra::WAIT_FAILED;

#[repr(C)]
#[cfg(target_arch = "x86")]
pub struct WSADATA {
    pub wVersion: libc::WORD,
    pub wHighVersion: libc::WORD,
    pub szDescription: [u8, ..WSADESCRIPTION_LEN + 1],
    pub szSystemStatus: [u8, ..WSASYS_STATUS_LEN + 1],
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
    pub szDescription: [u8, ..WSADESCRIPTION_LEN + 1],
    pub szSystemStatus: [u8, ..WSASYS_STATUS_LEN + 1],
}

pub type LPWSADATA = *mut WSADATA;

#[repr(C)]
pub struct WSANETWORKEVENTS {
    pub lNetworkEvents: libc::c_long,
    pub iErrorCode: [libc::c_int, ..FD_MAX_EVENTS],
}

pub type LPWSANETWORKEVENTS = *mut WSANETWORKEVENTS;

pub type WSAEVENT = libc::HANDLE;

#[repr(C)]
pub struct fd_set {
    fd_count: libc::c_uint,
    fd_array: [libc::SOCKET, ..FD_SETSIZE],
}

pub fn fd_set(set: &mut fd_set, s: libc::SOCKET) {
    set.fd_array[set.fd_count as uint] = s;
    set.fd_count += 1;
}

#[link(name = "ws2_32")]
extern "system" {
    pub fn WSAStartup(wVersionRequested: libc::WORD,
                      lpWSAData: LPWSADATA) -> libc::c_int;
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
    use std::intrinsics::{atomic_store_relaxed, transmute};
    use std::iter::Iterator;
    use libc::types::os::arch::extra::{LPCWSTR, HMODULE, LPCSTR, LPVOID};

    extern "system" {
        fn GetModuleHandleW(lpModuleName: LPCWSTR) -> HMODULE;
        fn GetProcAddress(hModule: HMODULE, lpProcName: LPCSTR) -> LPVOID;
    }

    // store_func() is idempotent, so using relaxed ordering for the atomics
    // should be enough.  This way, calling a function in this compatibility
    // layer (after it's loaded) shouldn't be any slower than a regular DLL
    // call.
    unsafe fn store_func(ptr: *mut uint, module: &str, symbol: &str, fallback: uint) {
        let module: Vec<u16> = module.utf16_units().collect();
        let module = module.append_one(0);
        symbol.with_c_str(|symbol| {
            let handle = GetModuleHandleW(module.as_ptr());
            let func: uint = transmute(GetProcAddress(handle, symbol));
            atomic_store_relaxed(ptr, if func == 0 {
                fallback
            } else {
                func
            })
        })
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
    /// Note that arguments unused by the fallback implementation should not be called `_` as
    /// they are used to be passed to the real function if available.
    macro_rules! compat_fn(
        ($module:ident::$symbol:ident($($argname:ident: $argtype:ty),*)
                                      -> $rettype:ty $fallback:block) => (
            #[inline(always)]
            pub unsafe fn $symbol($($argname: $argtype),*) -> $rettype {
                static mut ptr: extern "system" fn($($argname: $argtype),*) -> $rettype = thunk;

                extern "system" fn thunk($($argname: $argtype),*) -> $rettype {
                    unsafe {
                        ::io::c::compat::store_func(&mut ptr as *mut _ as *mut uint,
                                                    stringify!($module),
                                                    stringify!($symbol),
                                                    fallback as uint);
                        ::std::intrinsics::atomic_load_relaxed(&ptr)($($argname),*)
                    }
                }

                extern "system" fn fallback($($argname: $argtype),*) -> $rettype $fallback

                ::std::intrinsics::atomic_load_relaxed(&ptr)($($argname),*)
            }
        );

        ($module:ident::$symbol:ident($($argname:ident: $argtype:ty),*) $fallback:block) => (
            compat_fn!($module::$symbol($($argname: $argtype),*) -> () $fallback)
        )
    )

    /// Compatibility layer for functions in `kernel32.dll`
    ///
    /// Latest versions of Windows this is needed for:
    ///
    /// * `CreateSymbolicLinkW`: Windows XP, Windows Server 2003
    /// * `GetFinalPathNameByHandleW`: Windows XP, Windows Server 2003
    pub mod kernel32 {
        use libc::types::os::arch::extra::{DWORD, LPCWSTR, BOOLEAN, HANDLE};
        use libc::consts::os::extra::ERROR_CALL_NOT_IMPLEMENTED;

        extern "system" {
            fn SetLastError(dwErrCode: DWORD);
        }

        compat_fn!(kernel32::CreateSymbolicLinkW(_lpSymlinkFileName: LPCWSTR,
                                                 _lpTargetFileName: LPCWSTR,
                                                 _dwFlags: DWORD) -> BOOLEAN {
            unsafe { SetLastError(ERROR_CALL_NOT_IMPLEMENTED as DWORD); }
            0
        })

        compat_fn!(kernel32::GetFinalPathNameByHandleW(_hFile: HANDLE,
                                                       _lpszFilePath: LPCWSTR,
                                                       _cchFilePath: DWORD,
                                                       _dwFlags: DWORD) -> DWORD {
            unsafe { SetLastError(ERROR_CALL_NOT_IMPLEMENTED as DWORD); }
            0
        })
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
}
