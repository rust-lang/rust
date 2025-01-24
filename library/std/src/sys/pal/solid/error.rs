pub use self::itron::error::{ItronError as SolidError, expect_success};
use super::{abi, itron, net};
use crate::io::ErrorKind;

/// Describe the specified SOLID error code. Returns `None` if it's an
/// undefined error code.
///
/// The SOLID error codes are a superset of μITRON error codes.
pub fn error_name(er: abi::ER) -> Option<&'static str> {
    match er {
        // Success
        er if er >= 0 => None,
        er if er < abi::sockets::SOLID_NET_ERR_BASE => net::error_name(er),

        abi::SOLID_ERR_NOTFOUND => Some("not found"),
        abi::SOLID_ERR_NOTSUPPORTED => Some("not supported"),
        abi::SOLID_ERR_EBADF => Some("bad flags"),
        abi::SOLID_ERR_INVALIDCONTENT => Some("invalid content"),
        abi::SOLID_ERR_NOTUSED => Some("not used"),
        abi::SOLID_ERR_ALREADYUSED => Some("already used"),
        abi::SOLID_ERR_OUTOFBOUND => Some("out of bounds"),
        abi::SOLID_ERR_BADSEQUENCE => Some("bad sequence"),
        abi::SOLID_ERR_UNKNOWNDEVICE => Some("unknown device"),
        abi::SOLID_ERR_BUSY => Some("busy"),
        abi::SOLID_ERR_TIMEOUT => Some("operation timed out"),
        abi::SOLID_ERR_INVALIDACCESS => Some("invalid access"),
        abi::SOLID_ERR_NOTREADY => Some("not ready"),

        _ => itron::error::error_name(er),
    }
}

pub fn decode_error_kind(er: abi::ER) -> ErrorKind {
    match er {
        // Success
        er if er >= 0 => ErrorKind::Uncategorized,
        er if er < abi::sockets::SOLID_NET_ERR_BASE => net::decode_error_kind(er),

        abi::SOLID_ERR_NOTFOUND => ErrorKind::NotFound,
        abi::SOLID_ERR_NOTSUPPORTED => ErrorKind::Unsupported,
        abi::SOLID_ERR_EBADF => ErrorKind::InvalidInput,
        abi::SOLID_ERR_INVALIDCONTENT => ErrorKind::InvalidData,
        // abi::SOLID_ERR_NOTUSED
        // abi::SOLID_ERR_ALREADYUSED
        abi::SOLID_ERR_OUTOFBOUND => ErrorKind::InvalidInput,
        // abi::SOLID_ERR_BADSEQUENCE
        abi::SOLID_ERR_UNKNOWNDEVICE => ErrorKind::NotFound,
        // abi::SOLID_ERR_BUSY
        abi::SOLID_ERR_TIMEOUT => ErrorKind::TimedOut,
        // abi::SOLID_ERR_INVALIDACCESS
        // abi::SOLID_ERR_NOTREADY
        _ => itron::error::decode_error_kind(er),
    }
}
