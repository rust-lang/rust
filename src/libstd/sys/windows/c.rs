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
use libc::{c_uint, c_ulong};
use libc::{DWORD, BOOL, BOOLEAN, ERROR_CALL_NOT_IMPLEMENTED, LPVOID, HANDLE};
use libc::{LPCWSTR, LONG};
use ptr;

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
pub const ERROR_ENVVAR_NOT_FOUND: libc::c_int = 203;
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
pub const WSA_FLAG_OVERLAPPED: libc::DWORD = 0x01;
pub const WSA_FLAG_NO_HANDLE_INHERIT: libc::DWORD = 0x80;

pub const ERROR_NO_MORE_FILES: libc::DWORD = 18;
pub const TOKEN_READ: libc::DWORD = 0x20008;
pub const FILE_FLAG_OPEN_REPARSE_POINT: libc::DWORD = 0x00200000;
pub const FILE_FLAG_BACKUP_SEMANTICS: libc::DWORD = 0x02000000;
pub const MAXIMUM_REPARSE_DATA_BUFFER_SIZE: usize = 16 * 1024;
pub const FSCTL_GET_REPARSE_POINT: libc::DWORD = 0x900a8;
pub const IO_REPARSE_TAG_SYMLINK: libc::DWORD = 0xa000000c;
pub const IO_REPARSE_TAG_MOUNT_POINT: libc::DWORD = 0xa0000003;
pub const FSCTL_SET_REPARSE_POINT: libc::DWORD = 0x900a4;
pub const FSCTL_DELETE_REPARSE_POINT: libc::DWORD = 0x900ac;

pub const SYMBOLIC_LINK_FLAG_DIRECTORY: libc::DWORD = 0x1;

// Note that these are not actually HANDLEs, just values to pass to GetStdHandle
pub const STD_INPUT_HANDLE: libc::DWORD = -10i32 as libc::DWORD;
pub const STD_OUTPUT_HANDLE: libc::DWORD = -11i32 as libc::DWORD;
pub const STD_ERROR_HANDLE: libc::DWORD = -12i32 as libc::DWORD;

pub const HANDLE_FLAG_INHERIT: libc::DWORD = 0x00000001;

pub const PROGRESS_CONTINUE: libc::DWORD = 0;
pub const PROGRESS_CANCEL: libc::DWORD = 1;
pub const PROGRESS_STOP: libc::DWORD = 2;
pub const PROGRESS_QUIET: libc::DWORD = 3;

pub const TOKEN_ADJUST_PRIVILEGES: libc::DWORD = 0x0020;
pub const SE_PRIVILEGE_ENABLED: libc::DWORD = 2;

pub const EXCEPTION_CONTINUE_SEARCH: LONG = 0;
pub const EXCEPTION_MAXIMUM_PARAMETERS: usize = 15;
pub const EXCEPTION_STACK_OVERFLOW: DWORD = 0xc00000fd;

pub const ERROR_PATH_NOT_FOUND: libc::c_int = 3;

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
#[derive(Clone)]
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

#[repr(C)]
pub struct REPARSE_DATA_BUFFER {
    pub ReparseTag: libc::c_uint,
    pub ReparseDataLength: libc::c_ushort,
    pub Reserved: libc::c_ushort,
    pub rest: (),
}

#[repr(C)]
pub struct SYMBOLIC_LINK_REPARSE_BUFFER {
    pub SubstituteNameOffset: libc::c_ushort,
    pub SubstituteNameLength: libc::c_ushort,
    pub PrintNameOffset: libc::c_ushort,
    pub PrintNameLength: libc::c_ushort,
    pub Flags: libc::c_ulong,
    pub PathBuffer: libc::WCHAR,
}

pub type PCONDITION_VARIABLE = *mut CONDITION_VARIABLE;
pub type PSRWLOCK = *mut SRWLOCK;
pub type ULONG = c_ulong;
pub type ULONG_PTR = c_ulong;
pub type LPBOOL = *mut BOOL;

pub type LPPROGRESS_ROUTINE = ::option::Option<unsafe extern "system" fn(
    TotalFileSize: libc::LARGE_INTEGER,
    TotalBytesTransferred: libc::LARGE_INTEGER,
    StreamSize: libc::LARGE_INTEGER,
    StreamBytesTransferred: libc::LARGE_INTEGER,
    dwStreamNumber: DWORD,
    dwCallbackReason: DWORD,
    hSourceFile: HANDLE,
    hDestinationFile: HANDLE,
    lpData: LPVOID,
) -> DWORD>;

#[repr(C)]
pub struct CONDITION_VARIABLE { pub ptr: LPVOID }
#[repr(C)]
pub struct SRWLOCK { pub ptr: LPVOID }
#[repr(C)]
pub struct CRITICAL_SECTION {
    CriticalSectionDebug: LPVOID,
    LockCount: LONG,
    RecursionCount: LONG,
    OwningThread: HANDLE,
    LockSemaphore: HANDLE,
    SpinCount: ULONG_PTR
}

pub const CONDITION_VARIABLE_INIT: CONDITION_VARIABLE = CONDITION_VARIABLE {
    ptr: ptr::null_mut(),
};
pub const SRWLOCK_INIT: SRWLOCK = SRWLOCK { ptr: ptr::null_mut() };

#[repr(C)]
pub struct LUID {
    pub LowPart: libc::DWORD,
    pub HighPart: libc::c_long,
}

pub type PLUID = *mut LUID;

#[repr(C)]
pub struct TOKEN_PRIVILEGES {
    pub PrivilegeCount: libc::DWORD,
    pub Privileges: [LUID_AND_ATTRIBUTES; 1],
}

pub type PTOKEN_PRIVILEGES = *mut TOKEN_PRIVILEGES;

#[repr(C)]
pub struct LUID_AND_ATTRIBUTES {
    pub Luid: LUID,
    pub Attributes: libc::DWORD,
}

#[repr(C)]
pub struct REPARSE_MOUNTPOINT_DATA_BUFFER {
    pub ReparseTag: libc::DWORD,
    pub ReparseDataLength: libc::DWORD,
    pub Reserved: libc::WORD,
    pub ReparseTargetLength: libc::WORD,
    pub ReparseTargetMaximumLength: libc::WORD,
    pub Reserved1: libc::WORD,
    pub ReparseTarget: libc::WCHAR,
}

#[repr(C)]
pub struct EXCEPTION_RECORD {
    pub ExceptionCode: DWORD,
    pub ExceptionFlags: DWORD,
    pub ExceptionRecord: *mut EXCEPTION_RECORD,
    pub ExceptionAddress: LPVOID,
    pub NumberParameters: DWORD,
    pub ExceptionInformation: [LPVOID; EXCEPTION_MAXIMUM_PARAMETERS]
}

#[repr(C)]
pub struct EXCEPTION_POINTERS {
    pub ExceptionRecord: *mut EXCEPTION_RECORD,
    pub ContextRecord: LPVOID
}

pub type PVECTORED_EXCEPTION_HANDLER = extern "system"
        fn(ExceptionInfo: *mut EXCEPTION_POINTERS) -> LONG;

#[link(name = "ws2_32")]
#[link(name = "userenv")]
#[link(name = "shell32")]
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

    pub fn InitializeCriticalSection(CriticalSection: *mut CRITICAL_SECTION);
    pub fn EnterCriticalSection(CriticalSection: *mut CRITICAL_SECTION);
    pub fn TryEnterCriticalSection(CriticalSection: *mut CRITICAL_SECTION) -> BOOLEAN;
    pub fn LeaveCriticalSection(CriticalSection: *mut CRITICAL_SECTION);
    pub fn DeleteCriticalSection(CriticalSection: *mut CRITICAL_SECTION);

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
    pub fn GetTempPathW(nBufferLength: libc::DWORD,
                        lpBuffer: libc::LPCWSTR) -> libc::DWORD;
    pub fn OpenProcessToken(ProcessHandle: libc::HANDLE,
                            DesiredAccess: libc::DWORD,
                            TokenHandle: *mut libc::HANDLE) -> libc::BOOL;
    pub fn GetCurrentProcess() -> libc::HANDLE;
    pub fn GetStdHandle(which: libc::DWORD) -> libc::HANDLE;
    pub fn ExitProcess(uExitCode: libc::c_uint) -> !;
    pub fn DeviceIoControl(hDevice: libc::HANDLE,
                           dwIoControlCode: libc::DWORD,
                           lpInBuffer: libc::LPVOID,
                           nInBufferSize: libc::DWORD,
                           lpOutBuffer: libc::LPVOID,
                           nOutBufferSize: libc::DWORD,
                           lpBytesReturned: libc::LPDWORD,
                           lpOverlapped: libc::LPOVERLAPPED) -> libc::BOOL;
    pub fn CreatePipe(hReadPipe: libc::LPHANDLE,
                      hWritePipe: libc::LPHANDLE,
                      lpPipeAttributes: libc::LPSECURITY_ATTRIBUTES,
                      nSize: libc::DWORD) -> libc::BOOL;
    pub fn CreateThread(lpThreadAttributes: libc::LPSECURITY_ATTRIBUTES,
                        dwStackSize: libc::SIZE_T,
                        lpStartAddress: extern "system" fn(*mut libc::c_void)
                                                           -> libc::DWORD,
                        lpParameter: libc::LPVOID,
                        dwCreationFlags: libc::DWORD,
                        lpThreadId: libc::LPDWORD) -> libc::HANDLE;
    pub fn WaitForSingleObject(hHandle: libc::HANDLE,
                               dwMilliseconds: libc::DWORD) -> libc::DWORD;
    pub fn SwitchToThread() -> libc::BOOL;
    pub fn Sleep(dwMilliseconds: libc::DWORD);
    pub fn GetProcessId(handle: libc::HANDLE) -> libc::DWORD;
    pub fn GetUserProfileDirectoryW(hToken: libc::HANDLE,
                                    lpProfileDir: libc::LPCWSTR,
                                    lpcchSize: *mut libc::DWORD) -> libc::BOOL;
    pub fn SetHandleInformation(hObject: libc::HANDLE,
                                dwMask: libc::DWORD,
                                dwFlags: libc::DWORD) -> libc::BOOL;
    pub fn CopyFileExW(lpExistingFileName: libc::LPCWSTR,
                       lpNewFileName: libc::LPCWSTR,
                       lpProgressRoutine: LPPROGRESS_ROUTINE,
                       lpData: libc::LPVOID,
                       pbCancel: LPBOOL,
                       dwCopyFlags: libc::DWORD) -> libc::BOOL;
    pub fn LookupPrivilegeValueW(lpSystemName: libc::LPCWSTR,
                                 lpName: libc::LPCWSTR,
                                 lpLuid: PLUID) -> libc::BOOL;
    pub fn AdjustTokenPrivileges(TokenHandle: libc::HANDLE,
                                 DisableAllPrivileges: libc::BOOL,
                                 NewState: PTOKEN_PRIVILEGES,
                                 BufferLength: libc::DWORD,
                                 PreviousState: PTOKEN_PRIVILEGES,
                                 ReturnLength: *mut libc::DWORD) -> libc::BOOL;
    pub fn AddVectoredExceptionHandler(FirstHandler: ULONG,
                                       VectoredHandler: PVECTORED_EXCEPTION_HANDLER)
                                       -> LPVOID;
}

// Functions that aren't available on Windows XP, but we still use them and just
// provide some form of a fallback implementation.
compat_fn! {
    kernel32:

    pub fn CreateSymbolicLinkW(_lpSymlinkFileName: LPCWSTR,
                               _lpTargetFileName: LPCWSTR,
                               _dwFlags: DWORD) -> BOOLEAN {
        SetLastError(ERROR_CALL_NOT_IMPLEMENTED as DWORD); 0
    }
    pub fn GetFinalPathNameByHandleW(_hFile: HANDLE,
                                     _lpszFilePath: LPCWSTR,
                                     _cchFilePath: DWORD,
                                     _dwFlags: DWORD) -> DWORD {
        SetLastError(ERROR_CALL_NOT_IMPLEMENTED as DWORD); 0
    }
    pub fn SetThreadErrorMode(_dwNewMode: DWORD,
                              _lpOldMode: *mut DWORD) -> c_uint {
        SetLastError(ERROR_CALL_NOT_IMPLEMENTED as DWORD); 0
    }
    pub fn SetThreadStackGuarantee(_size: *mut c_ulong) -> BOOL {
        SetLastError(ERROR_CALL_NOT_IMPLEMENTED as DWORD); 0
    }
    pub fn SetFileInformationByHandle(_hFile: HANDLE,
                    _FileInformationClass: FILE_INFO_BY_HANDLE_CLASS,
                    _lpFileInformation: LPVOID,
                    _dwBufferSize: DWORD) -> BOOL {
        SetLastError(ERROR_CALL_NOT_IMPLEMENTED as DWORD); 0
    }
    pub fn SleepConditionVariableSRW(ConditionVariable: PCONDITION_VARIABLE,
                                     SRWLock: PSRWLOCK,
                                     dwMilliseconds: DWORD,
                                     Flags: ULONG) -> BOOL {
        panic!("condition variables not available")
    }
    pub fn WakeConditionVariable(ConditionVariable: PCONDITION_VARIABLE)
                                 -> () {
        panic!("condition variables not available")
    }
    pub fn WakeAllConditionVariable(ConditionVariable: PCONDITION_VARIABLE)
                                    -> () {
        panic!("condition variables not available")
    }
    pub fn AcquireSRWLockExclusive(SRWLock: PSRWLOCK) -> () {
        panic!("rwlocks not available")
    }
    pub fn AcquireSRWLockShared(SRWLock: PSRWLOCK) -> () {
        panic!("rwlocks not available")
    }
    pub fn ReleaseSRWLockExclusive(SRWLock: PSRWLOCK) -> () {
        panic!("rwlocks not available")
    }
    pub fn ReleaseSRWLockShared(SRWLock: PSRWLOCK) -> () {
        panic!("rwlocks not available")
    }
    pub fn TryAcquireSRWLockExclusive(SRWLock: PSRWLOCK) -> BOOLEAN {
        panic!("rwlocks not available")
    }
    pub fn TryAcquireSRWLockShared(SRWLock: PSRWLOCK) -> BOOLEAN {
        panic!("rwlocks not available")
    }
}
