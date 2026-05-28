use r_efi::efi::Status;

use crate::io;

pub fn errno() -> io::RawOsError {
    0
}

pub fn is_interrupted(_code: io::RawOsError) -> bool {
    false
}

pub fn decode_error_kind(code: io::RawOsError) -> io::ErrorKind {
    match Status::from_usize(code) {
        Status::ALREADY_STARTED
        | Status::COMPROMISED_DATA
        | Status::CONNECTION_FIN
        | Status::CRC_ERROR
        | Status::DEVICE_ERROR
        | Status::END_OF_MEDIA
        | Status::HTTP_ERROR
        | Status::ICMP_ERROR
        | Status::INCOMPATIBLE_VERSION
        | Status::LOAD_ERROR
        | Status::MEDIA_CHANGED
        | Status::NO_MAPPING
        | Status::NO_MEDIA
        | Status::NOT_STARTED
        | Status::PROTOCOL_ERROR
        | Status::PROTOCOL_UNREACHABLE
        | Status::TFTP_ERROR
        | Status::VOLUME_CORRUPTED => io::ErrorKind::Other,
        Status::BAD_BUFFER_SIZE | Status::INVALID_LANGUAGE => io::ErrorKind::InvalidData,
        Status::ABORTED => io::ErrorKind::ConnectionAborted,
        Status::ACCESS_DENIED => io::ErrorKind::PermissionDenied,
        Status::BUFFER_TOO_SMALL => io::ErrorKind::FileTooLarge,
        Status::CONNECTION_REFUSED => io::ErrorKind::ConnectionRefused,
        Status::CONNECTION_RESET => io::ErrorKind::ConnectionReset,
        Status::END_OF_FILE => io::ErrorKind::UnexpectedEof,
        Status::HOST_UNREACHABLE => io::ErrorKind::HostUnreachable,
        Status::INVALID_PARAMETER => io::ErrorKind::InvalidInput,
        Status::IP_ADDRESS_CONFLICT => io::ErrorKind::AddrInUse,
        Status::NETWORK_UNREACHABLE => io::ErrorKind::NetworkUnreachable,
        Status::NO_RESPONSE => io::ErrorKind::HostUnreachable,
        Status::NOT_FOUND => io::ErrorKind::NotFound,
        Status::NOT_READY => io::ErrorKind::ResourceBusy,
        Status::OUT_OF_RESOURCES => io::ErrorKind::OutOfMemory,
        Status::SECURITY_VIOLATION => io::ErrorKind::PermissionDenied,
        Status::TIMEOUT => io::ErrorKind::TimedOut,
        Status::UNSUPPORTED => io::ErrorKind::Unsupported,
        Status::VOLUME_FULL => io::ErrorKind::StorageFull,
        Status::WRITE_PROTECTED => io::ErrorKind::ReadOnlyFilesystem,
        _ => io::ErrorKind::Uncategorized,
    }
}

pub fn error_string(errno: io::RawOsError) -> String {
    // Keep the List in Alphabetical Order
    // The Messages are taken from UEFI Specification Appendix D - Status Codes
    #[rustfmt::skip]
    let msg = match Status::from_usize(errno) {
        Status::ABORTED => "The operation was aborted.",
        Status::ACCESS_DENIED => "Access was denied.",
        Status::ALREADY_STARTED => "The protocol has already been started.",
        Status::BAD_BUFFER_SIZE => "The buffer was not the proper size for the request.",
        Status::BUFFER_TOO_SMALL => "The buffer is not large enough to hold the requested data. The required buffer size is returned in the appropriate parameter when this error occurs.",
        Status::COMPROMISED_DATA => "The security status of the data is unknown or compromised and the data must be updated or replaced to restore a valid security status.",
        Status::CONNECTION_FIN => "The receiving operation fails because the communication peer has closed the connection and there is no more data in the receive buffer of the instance.",
        Status::CONNECTION_REFUSED => "The receiving or transmission operation fails because this connection is refused.",
        Status::CONNECTION_RESET => "The connect fails because the connection is reset either by instance itself or the communication peer.",
        Status::CRC_ERROR => "A CRC error was detected.",
        Status::DEVICE_ERROR => "The physical device reported an error while attempting the operation.",
        Status::END_OF_FILE => "The end of the file was reached.",
        Status::END_OF_MEDIA => "Beginning or end of media was reached",
        Status::HOST_UNREACHABLE => "The remote host is not reachable.",
        Status::HTTP_ERROR => "A HTTP error occurred during the network operation.",
        Status::ICMP_ERROR => "An ICMP error occurred during the network operation.",
        Status::INCOMPATIBLE_VERSION => "The function encountered an internal version that was incompatible with a version requested by the caller.",
        Status::INVALID_LANGUAGE => "The language specified was invalid.",
        Status::INVALID_PARAMETER => "A parameter was incorrect.",
        Status::IP_ADDRESS_CONFLICT => "There is an address conflict address allocation",
        Status::LOAD_ERROR => "The image failed to load.",
        Status::MEDIA_CHANGED => "The medium in the device has changed since the last access.",
        Status::NETWORK_UNREACHABLE => "The network containing the remote host is not reachable.",
        Status::NO_MAPPING => "A mapping to a device does not exist.",
        Status::NO_MEDIA => "The device does not contain any medium to perform the operation.",
        Status::NO_RESPONSE => "The server was not found or did not respond to the request.",
        Status::NOT_FOUND => "The item was not found.",
        Status::NOT_READY => "There is no data pending upon return.",
        Status::NOT_STARTED => "The protocol has not been started.",
        Status::OUT_OF_RESOURCES => "A resource has run out.",
        Status::PROTOCOL_ERROR => "A protocol error occurred during the network operation.",
        Status::PROTOCOL_UNREACHABLE => "An ICMP protocol unreachable error is received.",
        Status::SECURITY_VIOLATION => "The function was not performed due to a security violation.",
        Status::TFTP_ERROR => "A TFTP error occurred during the network operation.",
        Status::TIMEOUT => "The timeout time expired.",
        Status::UNSUPPORTED => "The operation is not supported.",
        Status::VOLUME_FULL => "There is no more space on the file system.",
        Status::VOLUME_CORRUPTED => "An inconstancy was detected on the file system causing the operating to fail.",
        Status::WRITE_PROTECTED => "The device cannot be written to.",
        _ => return format!("Status: {errno}"),
    };
    msg.to_owned()
}
