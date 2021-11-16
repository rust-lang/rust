//! C definitions used by libnative that don't belong in liblibc

#![allow(nonstandard_style)]
#![cfg_attr(test, allow(dead_code))]
#![unstable(issue = "none", feature = "windows_c")]

use crate::os::raw::NonZero_c_ulong;
use crate::os::raw::{c_char, c_int, c_long, c_longlong, c_uint, c_ulong, c_ushort};
use crate::ptr;

use libc::{c_void, size_t, wchar_t};

#[path = "c/errors.rs"] // c.rs is included from two places so we need to specify this
mod errors;
pub use errors::*;

pub use self::EXCEPTION_DISPOSITION::*;
pub use self::FILE_INFO_BY_HANDLE_CLASS::*;

pub type DWORD_PTR = ULONG_PTR;
pub type DWORD = c_ulong;
pub type NonZeroDWORD = NonZero_c_ulong;
pub type HANDLE = LPVOID;
pub type HINSTANCE = HANDLE;
pub type HMODULE = HINSTANCE;
pub type HRESULT = LONG;
pub type BOOL = c_int;
pub type BYTE = u8;
pub type BOOLEAN = BYTE;
pub type GROUP = c_uint;
pub type LARGE_INTEGER = c_longlong;
pub type LONG = c_long;
pub type UINT = c_uint;
pub type WCHAR = u16;
pub type USHORT = c_ushort;
pub type SIZE_T = usize;
pub type WORD = u16;
pub type CHAR = c_char;
pub type ULONG_PTR = usize;
pub type ULONG = c_ulong;
pub type NTSTATUS = LONG;
pub type ACCESS_MASK = DWORD;

pub type LPBOOL = *mut BOOL;
pub type LPBYTE = *mut BYTE;
pub type LPCSTR = *const CHAR;
pub type LPCWSTR = *const WCHAR;
pub type LPDWORD = *mut DWORD;
pub type LPHANDLE = *mut HANDLE;
pub type LPOVERLAPPED = *mut OVERLAPPED;
pub type LPPROCESS_INFORMATION = *mut PROCESS_INFORMATION;
pub type LPSECURITY_ATTRIBUTES = *mut SECURITY_ATTRIBUTES;
pub type LPSTARTUPINFO = *mut STARTUPINFO;
pub type LPVOID = *mut c_void;
pub type LPWCH = *mut WCHAR;
pub type LPWIN32_FIND_DATAW = *mut WIN32_FIND_DATAW;
pub type LPWSADATA = *mut WSADATA;
pub type LPWSAPROTOCOL_INFO = *mut WSAPROTOCOL_INFO;
pub type LPWSTR = *mut WCHAR;
pub type LPFILETIME = *mut FILETIME;
pub type LPSYSTEM_INFO = *mut SYSTEM_INFO;
pub type LPWSABUF = *mut WSABUF;
pub type LPWSAOVERLAPPED = *mut c_void;
pub type LPWSAOVERLAPPED_COMPLETION_ROUTINE = *mut c_void;

pub type PCONDITION_VARIABLE = *mut CONDITION_VARIABLE;
pub type PLARGE_INTEGER = *mut c_longlong;
pub type PSRWLOCK = *mut SRWLOCK;

pub type SOCKET = crate::os::windows::raw::SOCKET;
pub type socklen_t = c_int;
pub type ADDRESS_FAMILY = USHORT;

pub const TRUE: BOOL = 1;
pub const FALSE: BOOL = 0;

pub const CSTR_LESS_THAN: c_int = 1;
pub const CSTR_EQUAL: c_int = 2;
pub const CSTR_GREATER_THAN: c_int = 3;

pub const FILE_ATTRIBUTE_READONLY: DWORD = 0x1;
pub const FILE_ATTRIBUTE_DIRECTORY: DWORD = 0x10;
pub const FILE_ATTRIBUTE_REPARSE_POINT: DWORD = 0x400;

pub const FILE_SHARE_DELETE: DWORD = 0x4;
pub const FILE_SHARE_READ: DWORD = 0x1;
pub const FILE_SHARE_WRITE: DWORD = 0x2;

pub const CREATE_ALWAYS: DWORD = 2;
pub const CREATE_NEW: DWORD = 1;
pub const OPEN_ALWAYS: DWORD = 4;
pub const OPEN_EXISTING: DWORD = 3;
pub const TRUNCATE_EXISTING: DWORD = 5;

pub const FILE_WRITE_DATA: DWORD = 0x00000002;
pub const FILE_APPEND_DATA: DWORD = 0x00000004;
pub const FILE_WRITE_EA: DWORD = 0x00000010;
pub const FILE_WRITE_ATTRIBUTES: DWORD = 0x00000100;
pub const READ_CONTROL: DWORD = 0x00020000;
pub const SYNCHRONIZE: DWORD = 0x00100000;
pub const GENERIC_READ: DWORD = 0x80000000;
pub const GENERIC_WRITE: DWORD = 0x40000000;
pub const STANDARD_RIGHTS_WRITE: DWORD = READ_CONTROL;
pub const FILE_GENERIC_WRITE: DWORD = STANDARD_RIGHTS_WRITE
    | FILE_WRITE_DATA
    | FILE_WRITE_ATTRIBUTES
    | FILE_WRITE_EA
    | FILE_APPEND_DATA
    | SYNCHRONIZE;

pub const FILE_FLAG_OPEN_REPARSE_POINT: DWORD = 0x00200000;
pub const FILE_FLAG_BACKUP_SEMANTICS: DWORD = 0x02000000;
pub const SECURITY_SQOS_PRESENT: DWORD = 0x00100000;

pub const FIONBIO: c_ulong = 0x8004667e;

#[repr(C)]
#[derive(Copy)]
pub struct WIN32_FIND_DATAW {
    pub dwFileAttributes: DWORD,
    pub ftCreationTime: FILETIME,
    pub ftLastAccessTime: FILETIME,
    pub ftLastWriteTime: FILETIME,
    pub nFileSizeHigh: DWORD,
    pub nFileSizeLow: DWORD,
    pub dwReserved0: DWORD,
    pub dwReserved1: DWORD,
    pub cFileName: [wchar_t; 260], // #define MAX_PATH 260
    pub cAlternateFileName: [wchar_t; 14],
}
impl Clone for WIN32_FIND_DATAW {
    fn clone(&self) -> Self {
        *self
    }
}

pub const WSA_FLAG_OVERLAPPED: DWORD = 0x01;
pub const WSA_FLAG_NO_HANDLE_INHERIT: DWORD = 0x80;

pub const WSADESCRIPTION_LEN: usize = 256;
pub const WSASYS_STATUS_LEN: usize = 128;
pub const WSAPROTOCOL_LEN: DWORD = 255;
pub const INVALID_SOCKET: SOCKET = !0;

pub const MAX_PROTOCOL_CHAIN: DWORD = 7;

pub const MAXIMUM_REPARSE_DATA_BUFFER_SIZE: usize = 16 * 1024;
pub const FSCTL_GET_REPARSE_POINT: DWORD = 0x900a8;
pub const IO_REPARSE_TAG_SYMLINK: DWORD = 0xa000000c;
pub const IO_REPARSE_TAG_MOUNT_POINT: DWORD = 0xa0000003;
pub const SYMLINK_FLAG_RELATIVE: DWORD = 0x00000001;
pub const FSCTL_SET_REPARSE_POINT: DWORD = 0x900a4;

pub const SYMBOLIC_LINK_FLAG_DIRECTORY: DWORD = 0x1;
pub const SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE: DWORD = 0x2;

// Note that these are not actually HANDLEs, just values to pass to GetStdHandle
pub const STD_INPUT_HANDLE: DWORD = -10i32 as DWORD;
pub const STD_OUTPUT_HANDLE: DWORD = -11i32 as DWORD;
pub const STD_ERROR_HANDLE: DWORD = -12i32 as DWORD;

pub const PROGRESS_CONTINUE: DWORD = 0;

pub const E_NOTIMPL: HRESULT = 0x80004001u32 as HRESULT;

pub const INVALID_HANDLE_VALUE: HANDLE = !0 as HANDLE;

pub const FACILITY_NT_BIT: DWORD = 0x1000_0000;

pub const FORMAT_MESSAGE_FROM_SYSTEM: DWORD = 0x00001000;
pub const FORMAT_MESSAGE_FROM_HMODULE: DWORD = 0x00000800;
pub const FORMAT_MESSAGE_IGNORE_INSERTS: DWORD = 0x00000200;

pub const TLS_OUT_OF_INDEXES: DWORD = 0xFFFFFFFF;

pub const DLL_THREAD_DETACH: DWORD = 3;
pub const DLL_PROCESS_DETACH: DWORD = 0;

pub const INFINITE: DWORD = !0;

pub const DUPLICATE_SAME_ACCESS: DWORD = 0x00000002;

pub const CONDITION_VARIABLE_INIT: CONDITION_VARIABLE = CONDITION_VARIABLE { ptr: ptr::null_mut() };
pub const SRWLOCK_INIT: SRWLOCK = SRWLOCK { ptr: ptr::null_mut() };

pub const DETACHED_PROCESS: DWORD = 0x00000008;
pub const CREATE_NEW_PROCESS_GROUP: DWORD = 0x00000200;
pub const CREATE_UNICODE_ENVIRONMENT: DWORD = 0x00000400;
pub const STARTF_USESTDHANDLES: DWORD = 0x00000100;

pub const AF_INET: c_int = 2;
pub const AF_INET6: c_int = 23;
pub const SD_BOTH: c_int = 2;
pub const SD_RECEIVE: c_int = 0;
pub const SD_SEND: c_int = 1;
pub const SOCK_DGRAM: c_int = 2;
pub const SOCK_STREAM: c_int = 1;
pub const SOCKET_ERROR: c_int = -1;
pub const SOL_SOCKET: c_int = 0xffff;
pub const SO_LINGER: c_int = 0x0080;
pub const SO_RCVTIMEO: c_int = 0x1006;
pub const SO_SNDTIMEO: c_int = 0x1005;
pub const IPPROTO_IP: c_int = 0;
pub const IPPROTO_TCP: c_int = 6;
pub const IPPROTO_IPV6: c_int = 41;
pub const TCP_NODELAY: c_int = 0x0001;
pub const IP_TTL: c_int = 4;
pub const IPV6_V6ONLY: c_int = 27;
pub const SO_ERROR: c_int = 0x1007;
pub const SO_BROADCAST: c_int = 0x0020;
pub const IP_MULTICAST_LOOP: c_int = 11;
pub const IPV6_MULTICAST_LOOP: c_int = 11;
pub const IP_MULTICAST_TTL: c_int = 10;
pub const IP_ADD_MEMBERSHIP: c_int = 12;
pub const IP_DROP_MEMBERSHIP: c_int = 13;
pub const IPV6_ADD_MEMBERSHIP: c_int = 12;
pub const IPV6_DROP_MEMBERSHIP: c_int = 13;
pub const MSG_PEEK: c_int = 0x2;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct linger {
    pub l_onoff: c_ushort,
    pub l_linger: c_ushort,
}

#[repr(C)]
pub struct ip_mreq {
    pub imr_multiaddr: in_addr,
    pub imr_interface: in_addr,
}

#[repr(C)]
pub struct ipv6_mreq {
    pub ipv6mr_multiaddr: in6_addr,
    pub ipv6mr_interface: c_uint,
}

pub const VOLUME_NAME_DOS: DWORD = 0x0;
pub const MOVEFILE_REPLACE_EXISTING: DWORD = 1;

pub const FILE_BEGIN: DWORD = 0;
pub const FILE_CURRENT: DWORD = 1;
pub const FILE_END: DWORD = 2;

pub const WAIT_OBJECT_0: DWORD = 0x00000000;
pub const WAIT_TIMEOUT: DWORD = 258;
pub const WAIT_FAILED: DWORD = 0xFFFFFFFF;

pub const PIPE_ACCESS_INBOUND: DWORD = 0x00000001;
pub const PIPE_ACCESS_OUTBOUND: DWORD = 0x00000002;
pub const FILE_FLAG_FIRST_PIPE_INSTANCE: DWORD = 0x00080000;
pub const FILE_FLAG_OVERLAPPED: DWORD = 0x40000000;
pub const PIPE_WAIT: DWORD = 0x00000000;
pub const PIPE_TYPE_BYTE: DWORD = 0x00000000;
pub const PIPE_REJECT_REMOTE_CLIENTS: DWORD = 0x00000008;
pub const PIPE_READMODE_BYTE: DWORD = 0x00000000;

pub const FD_SETSIZE: usize = 64;

pub const STACK_SIZE_PARAM_IS_A_RESERVATION: DWORD = 0x00010000;

pub const STATUS_SUCCESS: NTSTATUS = 0x00000000;

pub const BCRYPT_USE_SYSTEM_PREFERRED_RNG: DWORD = 0x00000002;

#[repr(C)]
#[cfg(not(target_pointer_width = "64"))]
pub struct WSADATA {
    pub wVersion: WORD,
    pub wHighVersion: WORD,
    pub szDescription: [u8; WSADESCRIPTION_LEN + 1],
    pub szSystemStatus: [u8; WSASYS_STATUS_LEN + 1],
    pub iMaxSockets: u16,
    pub iMaxUdpDg: u16,
    pub lpVendorInfo: *mut u8,
}
#[repr(C)]
#[cfg(target_pointer_width = "64")]
pub struct WSADATA {
    pub wVersion: WORD,
    pub wHighVersion: WORD,
    pub iMaxSockets: u16,
    pub iMaxUdpDg: u16,
    pub lpVendorInfo: *mut u8,
    pub szDescription: [u8; WSADESCRIPTION_LEN + 1],
    pub szSystemStatus: [u8; WSASYS_STATUS_LEN + 1],
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct WSABUF {
    pub len: ULONG,
    pub buf: *mut CHAR,
}

#[repr(C)]
pub struct WSAPROTOCOL_INFO {
    pub dwServiceFlags1: DWORD,
    pub dwServiceFlags2: DWORD,
    pub dwServiceFlags3: DWORD,
    pub dwServiceFlags4: DWORD,
    pub dwProviderFlags: DWORD,
    pub ProviderId: GUID,
    pub dwCatalogEntryId: DWORD,
    pub ProtocolChain: WSAPROTOCOLCHAIN,
    pub iVersion: c_int,
    pub iAddressFamily: c_int,
    pub iMaxSockAddr: c_int,
    pub iMinSockAddr: c_int,
    pub iSocketType: c_int,
    pub iProtocol: c_int,
    pub iProtocolMaxOffset: c_int,
    pub iNetworkByteOrder: c_int,
    pub iSecurityScheme: c_int,
    pub dwMessageSize: DWORD,
    pub dwProviderReserved: DWORD,
    pub szProtocol: [u16; (WSAPROTOCOL_LEN as usize) + 1],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct WIN32_FILE_ATTRIBUTE_DATA {
    pub dwFileAttributes: DWORD,
    pub ftCreationTime: FILETIME,
    pub ftLastAccessTime: FILETIME,
    pub ftLastWriteTime: FILETIME,
    pub nFileSizeHigh: DWORD,
    pub nFileSizeLow: DWORD,
}

#[repr(C)]
#[allow(dead_code)] // we only use some variants
pub enum FILE_INFO_BY_HANDLE_CLASS {
    FileBasicInfo = 0,
    FileStandardInfo = 1,
    FileNameInfo = 2,
    FileRenameInfo = 3,
    FileDispositionInfo = 4,
    FileAllocationInfo = 5,
    FileEndOfFileInfo = 6,
    FileStreamInfo = 7,
    FileCompressionInfo = 8,
    FileAttributeTagInfo = 9,
    FileIdBothDirectoryInfo = 10,        // 0xA
    FileIdBothDirectoryRestartInfo = 11, // 0xB
    FileIoPriorityHintInfo = 12,         // 0xC
    FileRemoteProtocolInfo = 13,         // 0xD
    FileFullDirectoryInfo = 14,          // 0xE
    FileFullDirectoryRestartInfo = 15,   // 0xF
    FileStorageInfo = 16,                // 0x10
    FileAlignmentInfo = 17,              // 0x11
    FileIdInfo = 18,                     // 0x12
    FileIdExtdDirectoryInfo = 19,        // 0x13
    FileIdExtdDirectoryRestartInfo = 20, // 0x14
    MaximumFileInfoByHandlesClass,
}

#[repr(C)]
pub struct FILE_BASIC_INFO {
    pub CreationTime: LARGE_INTEGER,
    pub LastAccessTime: LARGE_INTEGER,
    pub LastWriteTime: LARGE_INTEGER,
    pub ChangeTime: LARGE_INTEGER,
    pub FileAttributes: DWORD,
}

#[repr(C)]
pub struct FILE_END_OF_FILE_INFO {
    pub EndOfFile: LARGE_INTEGER,
}

#[repr(C)]
pub struct REPARSE_DATA_BUFFER {
    pub ReparseTag: c_uint,
    pub ReparseDataLength: c_ushort,
    pub Reserved: c_ushort,
    pub rest: (),
}

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

pub type LPPROGRESS_ROUTINE = crate::option::Option<
    unsafe extern "system" fn(
        TotalFileSize: LARGE_INTEGER,
        TotalBytesTransferred: LARGE_INTEGER,
        StreamSize: LARGE_INTEGER,
        StreamBytesTransferred: LARGE_INTEGER,
        dwStreamNumber: DWORD,
        dwCallbackReason: DWORD,
        hSourceFile: HANDLE,
        hDestinationFile: HANDLE,
        lpData: LPVOID,
    ) -> DWORD,
>;

#[repr(C)]
pub struct CONDITION_VARIABLE {
    pub ptr: LPVOID,
}
#[repr(C)]
pub struct SRWLOCK {
    pub ptr: LPVOID,
}
#[repr(C)]
pub struct CRITICAL_SECTION {
    CriticalSectionDebug: LPVOID,
    LockCount: LONG,
    RecursionCount: LONG,
    OwningThread: HANDLE,
    LockSemaphore: HANDLE,
    SpinCount: ULONG_PTR,
}

#[repr(C)]
pub struct REPARSE_MOUNTPOINT_DATA_BUFFER {
    pub ReparseTag: DWORD,
    pub ReparseDataLength: DWORD,
    pub Reserved: WORD,
    pub ReparseTargetLength: WORD,
    pub ReparseTargetMaximumLength: WORD,
    pub Reserved1: WORD,
    pub ReparseTarget: WCHAR,
}

#[repr(C)]
pub struct GUID {
    pub Data1: DWORD,
    pub Data2: WORD,
    pub Data3: WORD,
    pub Data4: [BYTE; 8],
}

#[repr(C)]
pub struct WSAPROTOCOLCHAIN {
    pub ChainLen: c_int,
    pub ChainEntries: [DWORD; MAX_PROTOCOL_CHAIN as usize],
}

#[repr(C)]
pub struct SECURITY_ATTRIBUTES {
    pub nLength: DWORD,
    pub lpSecurityDescriptor: LPVOID,
    pub bInheritHandle: BOOL,
}

#[repr(C)]
pub struct PROCESS_INFORMATION {
    pub hProcess: HANDLE,
    pub hThread: HANDLE,
    pub dwProcessId: DWORD,
    pub dwThreadId: DWORD,
}

#[repr(C)]
pub struct STARTUPINFO {
    pub cb: DWORD,
    pub lpReserved: LPWSTR,
    pub lpDesktop: LPWSTR,
    pub lpTitle: LPWSTR,
    pub dwX: DWORD,
    pub dwY: DWORD,
    pub dwXSize: DWORD,
    pub dwYSize: DWORD,
    pub dwXCountChars: DWORD,
    pub dwYCountCharts: DWORD,
    pub dwFillAttribute: DWORD,
    pub dwFlags: DWORD,
    pub wShowWindow: WORD,
    pub cbReserved2: WORD,
    pub lpReserved2: LPBYTE,
    pub hStdInput: HANDLE,
    pub hStdOutput: HANDLE,
    pub hStdError: HANDLE,
}

#[repr(C)]
pub struct SOCKADDR {
    pub sa_family: ADDRESS_FAMILY,
    pub sa_data: [CHAR; 14],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct FILETIME {
    pub dwLowDateTime: DWORD,
    pub dwHighDateTime: DWORD,
}

#[repr(C)]
pub struct SYSTEM_INFO {
    pub wProcessorArchitecture: WORD,
    pub wReserved: WORD,
    pub dwPageSize: DWORD,
    pub lpMinimumApplicationAddress: LPVOID,
    pub lpMaximumApplicationAddress: LPVOID,
    pub dwActiveProcessorMask: DWORD_PTR,
    pub dwNumberOfProcessors: DWORD,
    pub dwProcessorType: DWORD,
    pub dwAllocationGranularity: DWORD,
    pub wProcessorLevel: WORD,
    pub wProcessorRevision: WORD,
}

#[repr(C)]
pub struct OVERLAPPED {
    pub Internal: *mut c_ulong,
    pub InternalHigh: *mut c_ulong,
    pub Offset: DWORD,
    pub OffsetHigh: DWORD,
    pub hEvent: HANDLE,
}

#[repr(C)]
#[allow(dead_code)] // we only use some variants
pub enum ADDRESS_MODE {
    AddrMode1616,
    AddrMode1632,
    AddrModeReal,
    AddrModeFlat,
}

#[repr(C)]
pub struct SOCKADDR_STORAGE_LH {
    pub ss_family: ADDRESS_FAMILY,
    pub __ss_pad1: [CHAR; 6],
    pub __ss_align: i64,
    pub __ss_pad2: [CHAR; 112],
}

#[repr(C)]
pub struct ADDRINFOA {
    pub ai_flags: c_int,
    pub ai_family: c_int,
    pub ai_socktype: c_int,
    pub ai_protocol: c_int,
    pub ai_addrlen: size_t,
    pub ai_canonname: *mut c_char,
    pub ai_addr: *mut SOCKADDR,
    pub ai_next: *mut ADDRINFOA,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct sockaddr_in {
    pub sin_family: ADDRESS_FAMILY,
    pub sin_port: USHORT,
    pub sin_addr: in_addr,
    pub sin_zero: [CHAR; 8],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct sockaddr_in6 {
    pub sin6_family: ADDRESS_FAMILY,
    pub sin6_port: USHORT,
    pub sin6_flowinfo: c_ulong,
    pub sin6_addr: in6_addr,
    pub sin6_scope_id: c_ulong,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct in_addr {
    pub s_addr: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct in6_addr {
    pub s6_addr: [u8; 16],
}

#[repr(C)]
#[derive(Copy, Clone)]
#[allow(dead_code)] // we only use some variants
pub enum EXCEPTION_DISPOSITION {
    ExceptionContinueExecution,
    ExceptionContinueSearch,
    ExceptionNestedException,
    ExceptionCollidedUnwind,
}

#[repr(C)]
#[derive(Copy)]
pub struct fd_set {
    pub fd_count: c_uint,
    pub fd_array: [SOCKET; FD_SETSIZE],
}

impl Clone for fd_set {
    fn clone(&self) -> fd_set {
        *self
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct timeval {
    pub tv_sec: c_long,
    pub tv_usec: c_long,
}

// Desktop specific functions & types
cfg_if::cfg_if! {
if #[cfg(not(target_vendor = "uwp"))] {
    pub const EXCEPTION_CONTINUE_SEARCH: LONG = 0;
    pub const EXCEPTION_STACK_OVERFLOW: DWORD = 0xc00000fd;
    pub const EXCEPTION_MAXIMUM_PARAMETERS: usize = 15;

    #[repr(C)]
    pub struct EXCEPTION_RECORD {
        pub ExceptionCode: DWORD,
        pub ExceptionFlags: DWORD,
        pub ExceptionRecord: *mut EXCEPTION_RECORD,
        pub ExceptionAddress: LPVOID,
        pub NumberParameters: DWORD,
        pub ExceptionInformation: [LPVOID; EXCEPTION_MAXIMUM_PARAMETERS],
    }

    pub enum CONTEXT {}

    #[repr(C)]
    pub struct EXCEPTION_POINTERS {
        pub ExceptionRecord: *mut EXCEPTION_RECORD,
        pub ContextRecord: *mut CONTEXT,
    }

    pub type PVECTORED_EXCEPTION_HANDLER =
        extern "system" fn(ExceptionInfo: *mut EXCEPTION_POINTERS) -> LONG;

    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct CONSOLE_READCONSOLE_CONTROL {
        pub nLength: ULONG,
        pub nInitialChars: ULONG,
        pub dwCtrlWakeupMask: ULONG,
        pub dwControlKeyState: ULONG,
    }

    pub type PCONSOLE_READCONSOLE_CONTROL = *mut CONSOLE_READCONSOLE_CONTROL;

    #[repr(C)]
    pub struct BY_HANDLE_FILE_INFORMATION {
        pub dwFileAttributes: DWORD,
        pub ftCreationTime: FILETIME,
        pub ftLastAccessTime: FILETIME,
        pub ftLastWriteTime: FILETIME,
        pub dwVolumeSerialNumber: DWORD,
        pub nFileSizeHigh: DWORD,
        pub nFileSizeLow: DWORD,
        pub nNumberOfLinks: DWORD,
        pub nFileIndexHigh: DWORD,
        pub nFileIndexLow: DWORD,
    }

    pub type LPBY_HANDLE_FILE_INFORMATION = *mut BY_HANDLE_FILE_INFORMATION;
    pub type LPCVOID = *const c_void;

    pub const HANDLE_FLAG_INHERIT: DWORD = 0x00000001;

    pub const TOKEN_READ: DWORD = 0x20008;

    #[link(name = "advapi32")]
    extern "system" {
        // Allowed but unused by UWP
        pub fn OpenProcessToken(
            ProcessHandle: HANDLE,
            DesiredAccess: DWORD,
            TokenHandle: *mut HANDLE,
        ) -> BOOL;
    }

    #[link(name = "userenv")]
    extern "system" {
        // Allowed but unused by UWP
        pub fn GetUserProfileDirectoryW(
            hToken: HANDLE,
            lpProfileDir: LPWSTR,
            lpcchSize: *mut DWORD,
        ) -> BOOL;
    }

    #[link(name = "kernel32")]
    extern "system" {
        // Functions forbidden when targeting UWP
        pub fn ReadConsoleW(
            hConsoleInput: HANDLE,
            lpBuffer: LPVOID,
            nNumberOfCharsToRead: DWORD,
            lpNumberOfCharsRead: LPDWORD,
            pInputControl: PCONSOLE_READCONSOLE_CONTROL,
        ) -> BOOL;

        pub fn WriteConsoleW(
            hConsoleOutput: HANDLE,
            lpBuffer: LPCVOID,
            nNumberOfCharsToWrite: DWORD,
            lpNumberOfCharsWritten: LPDWORD,
            lpReserved: LPVOID,
        ) -> BOOL;

        pub fn GetConsoleMode(hConsoleHandle: HANDLE, lpMode: LPDWORD) -> BOOL;
        // Allowed but unused by UWP
        pub fn GetFileInformationByHandle(
            hFile: HANDLE,
            lpFileInformation: LPBY_HANDLE_FILE_INFORMATION,
        ) -> BOOL;
        pub fn SetHandleInformation(hObject: HANDLE, dwMask: DWORD, dwFlags: DWORD) -> BOOL;
        pub fn AddVectoredExceptionHandler(
            FirstHandler: ULONG,
            VectoredHandler: PVECTORED_EXCEPTION_HANDLER,
        ) -> LPVOID;
        pub fn CreateHardLinkW(
            lpSymlinkFileName: LPCWSTR,
            lpTargetFileName: LPCWSTR,
            lpSecurityAttributes: LPSECURITY_ATTRIBUTES,
        ) -> BOOL;
        pub fn SetThreadStackGuarantee(_size: *mut c_ulong) -> BOOL;
        pub fn GetWindowsDirectoryW(lpBuffer: LPWSTR, uSize: UINT) -> UINT;
    }
}
}

// UWP specific functions & types
cfg_if::cfg_if! {
if #[cfg(target_vendor = "uwp")] {
    #[repr(C)]
    pub struct FILE_STANDARD_INFO {
        pub AllocationSize: LARGE_INTEGER,
        pub EndOfFile: LARGE_INTEGER,
        pub NumberOfLinks: DWORD,
        pub DeletePending: BOOLEAN,
        pub Directory: BOOLEAN,
    }

    #[link(name = "kernel32")]
    extern "system" {
        pub fn GetFileInformationByHandleEx(
            hFile: HANDLE,
            fileInfoClass: FILE_INFO_BY_HANDLE_CLASS,
            lpFileInformation: LPVOID,
            dwBufferSize: DWORD,
        ) -> BOOL;
    }
}
}

// Shared between Desktop & UWP

#[link(name = "kernel32")]
extern "system" {
    pub fn GetCurrentProcessId() -> DWORD;
    pub fn InitializeCriticalSection(CriticalSection: *mut CRITICAL_SECTION);
    pub fn EnterCriticalSection(CriticalSection: *mut CRITICAL_SECTION);
    pub fn TryEnterCriticalSection(CriticalSection: *mut CRITICAL_SECTION) -> BOOL;
    pub fn LeaveCriticalSection(CriticalSection: *mut CRITICAL_SECTION);
    pub fn DeleteCriticalSection(CriticalSection: *mut CRITICAL_SECTION);

    pub fn GetSystemDirectoryW(lpBuffer: LPWSTR, uSize: UINT) -> UINT;
    pub fn RemoveDirectoryW(lpPathName: LPCWSTR) -> BOOL;
    pub fn SetFileAttributesW(lpFileName: LPCWSTR, dwFileAttributes: DWORD) -> BOOL;
    pub fn SetLastError(dwErrCode: DWORD);
    pub fn GetCommandLineW() -> LPWSTR;
    pub fn GetTempPathW(nBufferLength: DWORD, lpBuffer: LPCWSTR) -> DWORD;
    pub fn GetCurrentProcess() -> HANDLE;
    pub fn GetCurrentThread() -> HANDLE;
    pub fn GetStdHandle(which: DWORD) -> HANDLE;
    pub fn ExitProcess(uExitCode: c_uint) -> !;
    pub fn DeviceIoControl(
        hDevice: HANDLE,
        dwIoControlCode: DWORD,
        lpInBuffer: LPVOID,
        nInBufferSize: DWORD,
        lpOutBuffer: LPVOID,
        nOutBufferSize: DWORD,
        lpBytesReturned: LPDWORD,
        lpOverlapped: LPOVERLAPPED,
    ) -> BOOL;
    pub fn CreateThread(
        lpThreadAttributes: LPSECURITY_ATTRIBUTES,
        dwStackSize: SIZE_T,
        lpStartAddress: extern "system" fn(*mut c_void) -> DWORD,
        lpParameter: LPVOID,
        dwCreationFlags: DWORD,
        lpThreadId: LPDWORD,
    ) -> HANDLE;
    pub fn WaitForSingleObject(hHandle: HANDLE, dwMilliseconds: DWORD) -> DWORD;
    pub fn SwitchToThread() -> BOOL;
    pub fn Sleep(dwMilliseconds: DWORD);
    pub fn GetProcessId(handle: HANDLE) -> DWORD;
    pub fn CopyFileExW(
        lpExistingFileName: LPCWSTR,
        lpNewFileName: LPCWSTR,
        lpProgressRoutine: LPPROGRESS_ROUTINE,
        lpData: LPVOID,
        pbCancel: LPBOOL,
        dwCopyFlags: DWORD,
    ) -> BOOL;
    pub fn FormatMessageW(
        flags: DWORD,
        lpSrc: LPVOID,
        msgId: DWORD,
        langId: DWORD,
        buf: LPWSTR,
        nsize: DWORD,
        args: *const c_void,
    ) -> DWORD;
    pub fn TlsAlloc() -> DWORD;
    pub fn TlsGetValue(dwTlsIndex: DWORD) -> LPVOID;
    pub fn TlsSetValue(dwTlsIndex: DWORD, lpTlsvalue: LPVOID) -> BOOL;
    pub fn GetLastError() -> DWORD;
    pub fn QueryPerformanceFrequency(lpFrequency: *mut LARGE_INTEGER) -> BOOL;
    pub fn QueryPerformanceCounter(lpPerformanceCount: *mut LARGE_INTEGER) -> BOOL;
    pub fn GetExitCodeProcess(hProcess: HANDLE, lpExitCode: LPDWORD) -> BOOL;
    pub fn TerminateProcess(hProcess: HANDLE, uExitCode: UINT) -> BOOL;
    pub fn CreateProcessW(
        lpApplicationName: LPCWSTR,
        lpCommandLine: LPWSTR,
        lpProcessAttributes: LPSECURITY_ATTRIBUTES,
        lpThreadAttributes: LPSECURITY_ATTRIBUTES,
        bInheritHandles: BOOL,
        dwCreationFlags: DWORD,
        lpEnvironment: LPVOID,
        lpCurrentDirectory: LPCWSTR,
        lpStartupInfo: LPSTARTUPINFO,
        lpProcessInformation: LPPROCESS_INFORMATION,
    ) -> BOOL;
    pub fn GetEnvironmentVariableW(n: LPCWSTR, v: LPWSTR, nsize: DWORD) -> DWORD;
    pub fn SetEnvironmentVariableW(n: LPCWSTR, v: LPCWSTR) -> BOOL;
    pub fn GetEnvironmentStringsW() -> LPWCH;
    pub fn FreeEnvironmentStringsW(env_ptr: LPWCH) -> BOOL;
    pub fn GetModuleFileNameW(hModule: HMODULE, lpFilename: LPWSTR, nSize: DWORD) -> DWORD;
    pub fn CreateDirectoryW(
        lpPathName: LPCWSTR,
        lpSecurityAttributes: LPSECURITY_ATTRIBUTES,
    ) -> BOOL;
    pub fn DeleteFileW(lpPathName: LPCWSTR) -> BOOL;
    pub fn GetCurrentDirectoryW(nBufferLength: DWORD, lpBuffer: LPWSTR) -> DWORD;
    pub fn SetCurrentDirectoryW(lpPathName: LPCWSTR) -> BOOL;
    pub fn DuplicateHandle(
        hSourceProcessHandle: HANDLE,
        hSourceHandle: HANDLE,
        hTargetProcessHandle: HANDLE,
        lpTargetHandle: LPHANDLE,
        dwDesiredAccess: DWORD,
        bInheritHandle: BOOL,
        dwOptions: DWORD,
    ) -> BOOL;
    pub fn ReadFile(
        hFile: HANDLE,
        lpBuffer: LPVOID,
        nNumberOfBytesToRead: DWORD,
        lpNumberOfBytesRead: LPDWORD,
        lpOverlapped: LPOVERLAPPED,
    ) -> BOOL;
    pub fn WriteFile(
        hFile: HANDLE,
        lpBuffer: LPVOID,
        nNumberOfBytesToWrite: DWORD,
        lpNumberOfBytesWritten: LPDWORD,
        lpOverlapped: LPOVERLAPPED,
    ) -> BOOL;
    pub fn CloseHandle(hObject: HANDLE) -> BOOL;
    pub fn MoveFileExW(lpExistingFileName: LPCWSTR, lpNewFileName: LPCWSTR, dwFlags: DWORD)
    -> BOOL;
    pub fn SetFilePointerEx(
        hFile: HANDLE,
        liDistanceToMove: LARGE_INTEGER,
        lpNewFilePointer: PLARGE_INTEGER,
        dwMoveMethod: DWORD,
    ) -> BOOL;
    pub fn FlushFileBuffers(hFile: HANDLE) -> BOOL;
    pub fn CreateFileW(
        lpFileName: LPCWSTR,
        dwDesiredAccess: DWORD,
        dwShareMode: DWORD,
        lpSecurityAttributes: LPSECURITY_ATTRIBUTES,
        dwCreationDisposition: DWORD,
        dwFlagsAndAttributes: DWORD,
        hTemplateFile: HANDLE,
    ) -> HANDLE;

    pub fn FindFirstFileW(fileName: LPCWSTR, findFileData: LPWIN32_FIND_DATAW) -> HANDLE;
    pub fn FindNextFileW(findFile: HANDLE, findFileData: LPWIN32_FIND_DATAW) -> BOOL;
    pub fn FindClose(findFile: HANDLE) -> BOOL;

    pub fn GetProcAddress(handle: HMODULE, name: LPCSTR) -> *mut c_void;
    pub fn GetModuleHandleA(lpModuleName: LPCSTR) -> HMODULE;
    pub fn GetModuleHandleW(lpModuleName: LPCWSTR) -> HMODULE;

    pub fn GetSystemTimeAsFileTime(lpSystemTimeAsFileTime: LPFILETIME);
    pub fn GetSystemInfo(lpSystemInfo: LPSYSTEM_INFO);

    pub fn CreateEventW(
        lpEventAttributes: LPSECURITY_ATTRIBUTES,
        bManualReset: BOOL,
        bInitialState: BOOL,
        lpName: LPCWSTR,
    ) -> HANDLE;
    pub fn WaitForMultipleObjects(
        nCount: DWORD,
        lpHandles: *const HANDLE,
        bWaitAll: BOOL,
        dwMilliseconds: DWORD,
    ) -> DWORD;
    pub fn CreateNamedPipeW(
        lpName: LPCWSTR,
        dwOpenMode: DWORD,
        dwPipeMode: DWORD,
        nMaxInstances: DWORD,
        nOutBufferSize: DWORD,
        nInBufferSize: DWORD,
        nDefaultTimeOut: DWORD,
        lpSecurityAttributes: LPSECURITY_ATTRIBUTES,
    ) -> HANDLE;
    pub fn CancelIo(handle: HANDLE) -> BOOL;
    pub fn GetOverlappedResult(
        hFile: HANDLE,
        lpOverlapped: LPOVERLAPPED,
        lpNumberOfBytesTransferred: LPDWORD,
        bWait: BOOL,
    ) -> BOOL;
    pub fn CreateSymbolicLinkW(
        lpSymlinkFileName: LPCWSTR,
        lpTargetFileName: LPCWSTR,
        dwFlags: DWORD,
    ) -> BOOLEAN;
    pub fn GetFinalPathNameByHandleW(
        hFile: HANDLE,
        lpszFilePath: LPCWSTR,
        cchFilePath: DWORD,
        dwFlags: DWORD,
    ) -> DWORD;
    pub fn SetFileInformationByHandle(
        hFile: HANDLE,
        FileInformationClass: FILE_INFO_BY_HANDLE_CLASS,
        lpFileInformation: LPVOID,
        dwBufferSize: DWORD,
    ) -> BOOL;
    pub fn SleepConditionVariableSRW(
        ConditionVariable: PCONDITION_VARIABLE,
        SRWLock: PSRWLOCK,
        dwMilliseconds: DWORD,
        Flags: ULONG,
    ) -> BOOL;

    pub fn WakeConditionVariable(ConditionVariable: PCONDITION_VARIABLE);
    pub fn WakeAllConditionVariable(ConditionVariable: PCONDITION_VARIABLE);

    pub fn AcquireSRWLockExclusive(SRWLock: PSRWLOCK);
    pub fn AcquireSRWLockShared(SRWLock: PSRWLOCK);
    pub fn ReleaseSRWLockExclusive(SRWLock: PSRWLOCK);
    pub fn ReleaseSRWLockShared(SRWLock: PSRWLOCK);
    pub fn TryAcquireSRWLockExclusive(SRWLock: PSRWLOCK) -> BOOLEAN;
    pub fn TryAcquireSRWLockShared(SRWLock: PSRWLOCK) -> BOOLEAN;

    pub fn CompareStringOrdinal(
        lpString1: LPCWSTR,
        cchCount1: c_int,
        lpString2: LPCWSTR,
        cchCount2: c_int,
        bIgnoreCase: BOOL,
    ) -> c_int;
    pub fn GetFullPathNameW(
        lpFileName: LPCWSTR,
        nBufferLength: DWORD,
        lpBuffer: LPWSTR,
        lpFilePart: *mut LPWSTR,
    ) -> DWORD;
}

#[link(name = "ws2_32")]
extern "system" {
    pub fn WSAStartup(wVersionRequested: WORD, lpWSAData: LPWSADATA) -> c_int;
    pub fn WSACleanup() -> c_int;
    pub fn WSAGetLastError() -> c_int;
    pub fn WSADuplicateSocketW(
        s: SOCKET,
        dwProcessId: DWORD,
        lpProtocolInfo: LPWSAPROTOCOL_INFO,
    ) -> c_int;
    pub fn WSASend(
        s: SOCKET,
        lpBuffers: LPWSABUF,
        dwBufferCount: DWORD,
        lpNumberOfBytesSent: LPDWORD,
        dwFlags: DWORD,
        lpOverlapped: LPWSAOVERLAPPED,
        lpCompletionRoutine: LPWSAOVERLAPPED_COMPLETION_ROUTINE,
    ) -> c_int;
    pub fn WSARecv(
        s: SOCKET,
        lpBuffers: LPWSABUF,
        dwBufferCount: DWORD,
        lpNumberOfBytesRecvd: LPDWORD,
        lpFlags: LPDWORD,
        lpOverlapped: LPWSAOVERLAPPED,
        lpCompletionRoutine: LPWSAOVERLAPPED_COMPLETION_ROUTINE,
    ) -> c_int;
    pub fn WSASocketW(
        af: c_int,
        kind: c_int,
        protocol: c_int,
        lpProtocolInfo: LPWSAPROTOCOL_INFO,
        g: GROUP,
        dwFlags: DWORD,
    ) -> SOCKET;
    pub fn ioctlsocket(s: SOCKET, cmd: c_long, argp: *mut c_ulong) -> c_int;
    pub fn closesocket(socket: SOCKET) -> c_int;
    pub fn recv(socket: SOCKET, buf: *mut c_void, len: c_int, flags: c_int) -> c_int;
    pub fn send(socket: SOCKET, buf: *const c_void, len: c_int, flags: c_int) -> c_int;
    pub fn recvfrom(
        socket: SOCKET,
        buf: *mut c_void,
        len: c_int,
        flags: c_int,
        addr: *mut SOCKADDR,
        addrlen: *mut c_int,
    ) -> c_int;
    pub fn sendto(
        socket: SOCKET,
        buf: *const c_void,
        len: c_int,
        flags: c_int,
        addr: *const SOCKADDR,
        addrlen: c_int,
    ) -> c_int;
    pub fn shutdown(socket: SOCKET, how: c_int) -> c_int;
    pub fn accept(socket: SOCKET, address: *mut SOCKADDR, address_len: *mut c_int) -> SOCKET;
    pub fn getsockopt(
        s: SOCKET,
        level: c_int,
        optname: c_int,
        optval: *mut c_char,
        optlen: *mut c_int,
    ) -> c_int;
    pub fn setsockopt(
        s: SOCKET,
        level: c_int,
        optname: c_int,
        optval: *const c_void,
        optlen: c_int,
    ) -> c_int;
    pub fn getsockname(socket: SOCKET, address: *mut SOCKADDR, address_len: *mut c_int) -> c_int;
    pub fn getpeername(socket: SOCKET, address: *mut SOCKADDR, address_len: *mut c_int) -> c_int;
    pub fn bind(socket: SOCKET, address: *const SOCKADDR, address_len: socklen_t) -> c_int;
    pub fn listen(socket: SOCKET, backlog: c_int) -> c_int;
    pub fn connect(socket: SOCKET, address: *const SOCKADDR, len: c_int) -> c_int;
    pub fn getaddrinfo(
        node: *const c_char,
        service: *const c_char,
        hints: *const ADDRINFOA,
        res: *mut *mut ADDRINFOA,
    ) -> c_int;
    pub fn freeaddrinfo(res: *mut ADDRINFOA);
    pub fn select(
        nfds: c_int,
        readfds: *mut fd_set,
        writefds: *mut fd_set,
        exceptfds: *mut fd_set,
        timeout: *const timeval,
    ) -> c_int;
}

#[link(name = "bcrypt")]
extern "system" {
    // >= Vista / Server 2008
    // https://docs.microsoft.com/en-us/windows/win32/api/bcrypt/nf-bcrypt-bcryptgenrandom
    pub fn BCryptGenRandom(
        hAlgorithm: LPVOID,
        pBuffer: *mut u8,
        cbBuffer: ULONG,
        dwFlags: ULONG,
    ) -> NTSTATUS;
}

// Functions that aren't available on every version of Windows that we support,
// but we still use them and just provide some form of a fallback implementation.
compat_fn! {
    "kernel32":

    // >= Win10 1607
    // https://docs.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setthreaddescription
    pub fn SetThreadDescription(hThread: HANDLE,
                                lpThreadDescription: LPCWSTR) -> HRESULT {
        SetLastError(ERROR_CALL_NOT_IMPLEMENTED as DWORD); E_NOTIMPL
    }

    // >= Win8 / Server 2012
    // https://docs.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-getsystemtimepreciseasfiletime
    pub fn GetSystemTimePreciseAsFileTime(lpSystemTimeAsFileTime: LPFILETIME)
                                          -> () {
        GetSystemTimeAsFileTime(lpSystemTimeAsFileTime)
    }
}

compat_fn! {
    "api-ms-win-core-synch-l1-2-0":

    // >= Windows 8 / Server 2012
    // https://docs.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-waitonaddress
    pub fn WaitOnAddress(
        Address: LPVOID,
        CompareAddress: LPVOID,
        AddressSize: SIZE_T,
        dwMilliseconds: DWORD
    ) -> BOOL {
        panic!("WaitOnAddress not available")
    }
    pub fn WakeByAddressSingle(Address: LPVOID) -> () {
        // If this api is unavailable, there cannot be anything waiting, because
        // WaitOnAddress would've panicked. So it's fine to do nothing here.
    }
}

compat_fn! {
    "ntdll":
    pub fn NtCreateKeyedEvent(
        KeyedEventHandle: LPHANDLE,
        DesiredAccess: ACCESS_MASK,
        ObjectAttributes: LPVOID,
        Flags: ULONG
    ) -> NTSTATUS {
        panic!("keyed events not available")
    }
    pub fn NtReleaseKeyedEvent(
        EventHandle: HANDLE,
        Key: LPVOID,
        Alertable: BOOLEAN,
        Timeout: PLARGE_INTEGER
    ) -> NTSTATUS {
        panic!("keyed events not available")
    }
    pub fn NtWaitForKeyedEvent(
        EventHandle: HANDLE,
        Key: LPVOID,
        Alertable: BOOLEAN,
        Timeout: PLARGE_INTEGER
    ) -> NTSTATUS {
        panic!("keyed events not available")
    }
}
