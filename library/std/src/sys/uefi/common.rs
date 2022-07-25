use crate::io;

pub(crate) fn status_to_io_error(s: r_efi::efi::Status) -> crate::io::Error {
    use io::ErrorKind;
    use r_efi::efi::Status;

    match s {
        Status::INVALID_PARAMETER => {
            io::Error::new(ErrorKind::InvalidInput, "EFI_INVALID_PARAMETER")
        }
        Status::UNSUPPORTED => io::Error::new(ErrorKind::Unsupported, "EFI_UNSUPPORTED"),
        Status::BAD_BUFFER_SIZE => io::Error::new(ErrorKind::InvalidData, "EFI_BAD_BUFFER_SIZE"),
        Status::INVALID_LANGUAGE => io::Error::new(ErrorKind::InvalidData, "EFI_INVALID_LANGUAGE"),
        Status::CRC_ERROR => io::Error::new(ErrorKind::InvalidData, "EFI_CRC_ERROR"),
        Status::BUFFER_TOO_SMALL => io::Error::new(ErrorKind::FileTooLarge, "EFI_BUFFER_TOO_SMALL"),
        Status::NOT_READY => io::Error::new(ErrorKind::ResourceBusy, "EFI_NOT_READY"),
        Status::WRITE_PROTECTED => {
            io::Error::new(ErrorKind::ReadOnlyFilesystem, "EFI_WRITE_PROTECTED")
        }
        Status::VOLUME_FULL => io::Error::new(ErrorKind::StorageFull, "EFI_VOLUME_FULL"),
        Status::MEDIA_CHANGED => {
            io::Error::new(ErrorKind::StaleNetworkFileHandle, "EFI_MEDIA_CHANGED")
        }
        Status::NOT_FOUND => io::Error::new(ErrorKind::NotFound, "EFI_NOT_FOUND"),
        Status::ACCESS_DENIED => io::Error::new(ErrorKind::PermissionDenied, "EFI_ACCESS_DENIED"),
        Status::SECURITY_VIOLATION => {
            io::Error::new(ErrorKind::PermissionDenied, "EFI_SECURITY_VIOLATION")
        }
        Status::NO_RESPONSE => io::Error::new(ErrorKind::HostUnreachable, "EFI_NO_RESPONSE"),
        Status::TIMEOUT => io::Error::new(ErrorKind::TimedOut, "EFI_TIMEOUT"),
        Status::END_OF_FILE => io::Error::new(ErrorKind::UnexpectedEof, "EFI_END_OF_FILE"),
        Status::IP_ADDRESS_CONFLICT => {
            io::Error::new(ErrorKind::AddrInUse, "EFI_IP_ADDRESS_CONFLICT")
        }
        Status::HTTP_ERROR => io::Error::new(ErrorKind::NetworkUnreachable, "EFI_HTTP_ERROR"),
        _ => io::Error::new(ErrorKind::Uncategorized, format!("Status: {}", s.as_usize())),
    }
}
