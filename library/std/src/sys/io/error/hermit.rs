use crate::io;

pub fn errno() -> i32 {
    unsafe { hermit_abi::get_errno() }
}

#[inline]
pub fn is_interrupted(errno: i32) -> bool {
    errno == hermit_abi::errno::EINTR
}

pub fn decode_error_kind(errno: i32) -> io::ErrorKind {
    match errno {
        hermit_abi::errno::EACCES => io::ErrorKind::PermissionDenied,
        hermit_abi::errno::EADDRINUSE => io::ErrorKind::AddrInUse,
        hermit_abi::errno::EADDRNOTAVAIL => io::ErrorKind::AddrNotAvailable,
        hermit_abi::errno::EAGAIN => io::ErrorKind::WouldBlock,
        hermit_abi::errno::ECONNABORTED => io::ErrorKind::ConnectionAborted,
        hermit_abi::errno::ECONNREFUSED => io::ErrorKind::ConnectionRefused,
        hermit_abi::errno::ECONNRESET => io::ErrorKind::ConnectionReset,
        hermit_abi::errno::EEXIST => io::ErrorKind::AlreadyExists,
        hermit_abi::errno::EINTR => io::ErrorKind::Interrupted,
        hermit_abi::errno::EINVAL => io::ErrorKind::InvalidInput,
        hermit_abi::errno::ENOENT => io::ErrorKind::NotFound,
        hermit_abi::errno::ENOTCONN => io::ErrorKind::NotConnected,
        hermit_abi::errno::EPERM => io::ErrorKind::PermissionDenied,
        hermit_abi::errno::EPIPE => io::ErrorKind::BrokenPipe,
        hermit_abi::errno::ETIMEDOUT => io::ErrorKind::TimedOut,
        _ => io::ErrorKind::Uncategorized,
    }
}

pub fn error_string(errno: i32) -> String {
    hermit_abi::error_string(errno).to_string()
}
