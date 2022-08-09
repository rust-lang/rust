use r_efi::efi::Status;

use crate::io::{self, ErrorKind};

pub(crate) fn status_to_io_error(s: r_efi::efi::Status) -> io::Error {
    match s {
        Status::INVALID_PARAMETER => {
            io::error::const_io_error!(ErrorKind::InvalidInput, "EFI_INVALID_PARAMETER")
        }
        Status::UNSUPPORTED => {
            io::error::const_io_error!(ErrorKind::Unsupported, "EFI_UNSUPPORTED")
        }
        Status::BAD_BUFFER_SIZE => {
            io::error::const_io_error!(ErrorKind::InvalidData, "EFI_BAD_BUFFER_SIZE")
        }
        Status::INVALID_LANGUAGE => {
            io::error::const_io_error!(ErrorKind::InvalidData, "EFI_INVALID_LANGUAGE")
        }
        Status::CRC_ERROR => io::error::const_io_error!(ErrorKind::InvalidData, "EFI_CRC_ERROR"),
        Status::BUFFER_TOO_SMALL => {
            io::error::const_io_error!(ErrorKind::FileTooLarge, "EFI_BUFFER_TOO_SMALL")
        }
        Status::NOT_READY => io::error::const_io_error!(ErrorKind::ResourceBusy, "EFI_NOT_READY"),
        Status::WRITE_PROTECTED => {
            io::error::const_io_error!(ErrorKind::ReadOnlyFilesystem, "EFI_WRITE_PROTECTED")
        }
        Status::VOLUME_FULL => {
            io::error::const_io_error!(ErrorKind::StorageFull, "EFI_VOLUME_FULL")
        }
        Status::MEDIA_CHANGED => {
            io::error::const_io_error!(ErrorKind::StaleNetworkFileHandle, "EFI_MEDIA_CHANGED")
        }
        Status::NOT_FOUND => io::error::const_io_error!(ErrorKind::NotFound, "EFI_NOT_FOUND"),
        Status::ACCESS_DENIED => {
            io::error::const_io_error!(ErrorKind::PermissionDenied, "EFI_ACCESS_DENIED")
        }
        Status::SECURITY_VIOLATION => {
            io::error::const_io_error!(ErrorKind::PermissionDenied, "EFI_SECURITY_VIOLATION")
        }
        Status::NO_RESPONSE => {
            io::error::const_io_error!(ErrorKind::HostUnreachable, "EFI_NO_RESPONSE")
        }
        Status::TIMEOUT => io::error::const_io_error!(ErrorKind::TimedOut, "EFI_TIMEOUT"),
        Status::END_OF_FILE => {
            io::error::const_io_error!(ErrorKind::UnexpectedEof, "EFI_END_OF_FILE")
        }
        Status::IP_ADDRESS_CONFLICT => {
            io::error::const_io_error!(ErrorKind::AddrInUse, "EFI_IP_ADDRESS_CONFLICT")
        }
        Status::HTTP_ERROR => {
            io::error::const_io_error!(ErrorKind::NetworkUnreachable, "EFI_HTTP_ERROR")
        }
        _ => io::Error::new(ErrorKind::Uncategorized, format!("Status: {}", s.as_usize())),
    }
}
