use fortanix_sgx_abi::{Error, RESULT_SUCCESS};

use crate::io;

pub fn errno() -> i32 {
    RESULT_SUCCESS
}

#[inline]
pub fn is_interrupted(code: i32) -> bool {
    code == fortanix_sgx_abi::Error::Interrupted as _
}

pub fn decode_error_kind(code: i32) -> io::ErrorKind {
    // FIXME: not sure how to make sure all variants of Error are covered
    if code == Error::NotFound as _ {
        io::ErrorKind::NotFound
    } else if code == Error::PermissionDenied as _ {
        io::ErrorKind::PermissionDenied
    } else if code == Error::ConnectionRefused as _ {
        io::ErrorKind::ConnectionRefused
    } else if code == Error::ConnectionReset as _ {
        io::ErrorKind::ConnectionReset
    } else if code == Error::ConnectionAborted as _ {
        io::ErrorKind::ConnectionAborted
    } else if code == Error::NotConnected as _ {
        io::ErrorKind::NotConnected
    } else if code == Error::AddrInUse as _ {
        io::ErrorKind::AddrInUse
    } else if code == Error::AddrNotAvailable as _ {
        io::ErrorKind::AddrNotAvailable
    } else if code == Error::BrokenPipe as _ {
        io::ErrorKind::BrokenPipe
    } else if code == Error::AlreadyExists as _ {
        io::ErrorKind::AlreadyExists
    } else if code == Error::WouldBlock as _ {
        io::ErrorKind::WouldBlock
    } else if code == Error::InvalidInput as _ {
        io::ErrorKind::InvalidInput
    } else if code == Error::InvalidData as _ {
        io::ErrorKind::InvalidData
    } else if code == Error::TimedOut as _ {
        io::ErrorKind::TimedOut
    } else if code == Error::WriteZero as _ {
        io::ErrorKind::WriteZero
    } else if code == Error::Interrupted as _ {
        io::ErrorKind::Interrupted
    } else if code == Error::Other as _ {
        io::ErrorKind::Uncategorized
    } else if code == Error::UnexpectedEof as _ {
        io::ErrorKind::UnexpectedEof
    } else {
        io::ErrorKind::Uncategorized
    }
}

pub fn error_string(errno: i32) -> String {
    if errno == RESULT_SUCCESS {
        "operation successful".into()
    } else if ((Error::UserRangeStart as _)..=(Error::UserRangeEnd as _)).contains(&errno) {
        format!("user-specified error {errno:08x}")
    } else {
        decode_error_kind(errno).as_str().into()
    }
}
