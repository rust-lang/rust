use string::String;
use borrow::Cow;
use sys::windows::c;
use io::ErrorKind;
use libc;
use ptr;

pub use sys::common::error::{Result, expect_last_result, expect_last_error};

pub enum Error {
    Code(i32),
    NoStdioHandle,
    InvalidEncoding,
    NotSymlink,
    Unknown
}
pub struct ErrorString(String);

impl Error {
    pub fn from_code(code: i32) -> Self {
        Error::Code(code)
    }

    pub fn last_error() -> Option<Self> {
        Some(Error::Code(unsafe { libc::GetLastError() as i32 }))
    }

    pub fn default() -> Self {
        Error::Unknown
    }

    pub fn description(&self) -> ErrorString {
        match *self {
            Error::Code(errno) => error_description(errno),
            Error::InvalidEncoding => ErrorString("text was not valid unicode".into()),
            Error::NoStdioHandle => ErrorString("no stdio handle available for this process".into()),
            Error::NotSymlink => ErrorString("not a symlink".into()),
            Error::Unknown => ErrorString("unknown error".into()),
        }
    }

    pub fn kind(&self) -> ErrorKind {
        match *self {
            Error::Code(errno) => match errno as libc::c_int {
                libc::ERROR_ACCESS_DENIED => ErrorKind::PermissionDenied,
                libc::ERROR_ALREADY_EXISTS => ErrorKind::AlreadyExists,
                libc::ERROR_BROKEN_PIPE => ErrorKind::BrokenPipe,
                libc::ERROR_FILE_NOT_FOUND => ErrorKind::NotFound,
                c::ERROR_PATH_NOT_FOUND => ErrorKind::NotFound,
                libc::ERROR_NO_DATA => ErrorKind::BrokenPipe,
                libc::ERROR_OPERATION_ABORTED => ErrorKind::TimedOut,

                libc::WSAEACCES => ErrorKind::PermissionDenied,
                libc::WSAEADDRINUSE => ErrorKind::AddrInUse,
                libc::WSAEADDRNOTAVAIL => ErrorKind::AddrNotAvailable,
                libc::WSAECONNABORTED => ErrorKind::ConnectionAborted,
                libc::WSAECONNREFUSED => ErrorKind::ConnectionRefused,
                libc::WSAECONNRESET => ErrorKind::ConnectionReset,
                libc::WSAEINVAL => ErrorKind::InvalidInput,
                libc::WSAENOTCONN => ErrorKind::NotConnected,
                libc::WSAEWOULDBLOCK => ErrorKind::WouldBlock,
                libc::WSAETIMEDOUT => ErrorKind::TimedOut,

                _ => ErrorKind::Other,
            },
            Error::InvalidEncoding => ErrorKind::InvalidData,
            Error::NoStdioHandle | Error::NotSymlink | Error::Unknown => ErrorKind::Other,
        }
    }

    pub fn code(&self) -> i32 {
        match *self {
            Error::Code(code) => code,
            _ => 0,
        }
    }
}

impl ErrorString {
    pub fn to_string_lossy(&self) -> Cow<str> {
        (&*self.0).into()
    }
}

fn error_description(errnum: i32) -> ErrorString {
    use libc::types::os::arch::extra::DWORD;
    use libc::types::os::arch::extra::LPWSTR;
    use libc::types::os::arch::extra::LPVOID;
    use libc::types::os::arch::extra::WCHAR;

    #[link_name = "kernel32"]
    extern "system" {
        fn FormatMessageW(flags: DWORD,
                          lpSrc: LPVOID,
                          msgId: DWORD,
                          langId: DWORD,
                          buf: LPWSTR,
                          nsize: DWORD,
                          args: *const libc::c_void)
                          -> DWORD;
    }

    const FORMAT_MESSAGE_FROM_SYSTEM: DWORD = 0x00001000;
    const FORMAT_MESSAGE_IGNORE_INSERTS: DWORD = 0x00000200;

    // This value is calculated from the macro
    // MAKELANGID(LANG_SYSTEM_DEFAULT, SUBLANG_SYS_DEFAULT)
    let langId = 0x0800 as DWORD;

    let mut buf = [0 as WCHAR; 2048];

    unsafe {
        let res = FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM |
                                 FORMAT_MESSAGE_IGNORE_INSERTS,
                                 ptr::null_mut(),
                                 errnum as DWORD,
                                 langId,
                                 buf.as_mut_ptr(),
                                 buf.len() as DWORD,
                                 ptr::null()) as usize;
        if res == 0 {
            // Sometimes FormatMessageW can fail e.g. system doesn't like langId,
            let fm_err = expect_last_error().code();
            return ErrorString(format!("OS Error {} (FormatMessageW() returned error {})",
                           errnum, fm_err));
        }

        ErrorString(match String::from_utf16(&buf[..res]) {
            Ok(mut msg) => {
                // Trim trailing CRLF inserted by FormatMessageW
                let len = msg.trim_right().len();
                msg.truncate(len);
                msg
            },
            Err(..) => format!("OS Error {} (FormatMessageW() returned \
                                invalid UTF-16)", errnum),
        })
    }
}
