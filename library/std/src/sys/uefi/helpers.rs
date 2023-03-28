//! Contains most of the shared UEFI specific stuff. Some of this might be moved to `std::os::uefi`
//! if needed but no point in adding extra public API when there is not Std support for UEFI in the
//! first place
//!
//! Some Nomenclature
//! * Protocol:
//! - Protocols serve to enable communication between separately built modules, including drivers.
//! - Every protocol has a GUID associated with it. The GUID serves as the name for the protocol.
//! - Protocols are produced and consumed.
//! - More information about protocols can be found [here](https://edk2-docs.gitbook.io/edk-ii-uefi-driver-writer-s-guide/3_foundation/36_protocols_and_handles)

use r_efi::efi::{self, Guid};

use crate::mem::{size_of, MaybeUninit};
use crate::os::uefi;
use crate::ptr::NonNull;
use crate::{
    io::{self, const_io_error},
    os::uefi::env::boot_services,
};

const BOOT_SERVICES_UNAVAILABLE: io::Error =
    const_io_error!(io::ErrorKind::Other, "Boot Services are no longer available");

/// Locate Handles with a particular Protocol GUID
/// Implemented using `EFI_BOOT_SERVICES.LocateHandles()`
///
/// Returns an array of [Handles](r_efi::efi::Handle) that support a specified protocol.
pub(crate) fn locate_handles(mut guid: Guid) -> io::Result<Vec<NonNull<crate::ffi::c_void>>> {
    fn inner(
        guid: &mut Guid,
        boot_services: NonNull<r_efi::efi::BootServices>,
        buf_size: &mut usize,
        buf: *mut r_efi::efi::Handle,
    ) -> io::Result<()> {
        let r = unsafe {
            ((*boot_services.as_ptr()).locate_handle)(
                r_efi::efi::BY_PROTOCOL,
                guid,
                crate::ptr::null_mut(),
                buf_size,
                buf,
            )
        };

        if r.is_error() { Err(status_to_io_error(r)) } else { Ok(()) }
    }

    let boot_services = boot_services().ok_or(BOOT_SERVICES_UNAVAILABLE)?.cast();
    let mut buf_len = 0usize;

    // This should always fail since the size of buffer is 0. This call should update the buf_len
    // variable with the required buffer length
    match inner(&mut guid, boot_services, &mut buf_len, crate::ptr::null_mut()) {
        Ok(()) => unreachable!(),
        Err(e) => match e.kind() {
            io::ErrorKind::FileTooLarge => {}
            _ => return Err(e),
        },
    }

    // The returned buf_len is in bytes
    let mut buf: Vec<r_efi::efi::Handle> =
        Vec::with_capacity(buf_len / size_of::<r_efi::efi::Handle>());
    match inner(&mut guid, boot_services, &mut buf_len, buf.as_mut_ptr()) {
        Ok(()) => {
            // This is safe because the call will succeed only if buf_len >= required length.
            // Also, on success, the `buf_len` is updated with the size of bufferv (in bytes) written
            unsafe { buf.set_len(buf_len / size_of::<r_efi::efi::Handle>()) };
            Ok(buf.into_iter().filter_map(|x| NonNull::new(x)).collect())
        }
        Err(e) => Err(e),
    }
}

/// Open Protocol on a handle.
/// Internally just a call to `EFI_BOOT_SERVICES.OpenProtocol()`.
///
/// Queries a handle to determine if it supports a specified protocol. If the protocol is
/// supported by the handle, it opens the protocol on behalf of the calling agent.
pub(crate) fn open_protocol<T>(
    handle: NonNull<crate::ffi::c_void>,
    mut protocol_guid: Guid,
) -> io::Result<NonNull<T>> {
    let boot_services: NonNull<efi::BootServices> =
        boot_services().ok_or(BOOT_SERVICES_UNAVAILABLE)?.cast();
    let system_handle = uefi::env::image_handle();
    let mut protocol: MaybeUninit<*mut T> = MaybeUninit::uninit();

    let r = unsafe {
        ((*boot_services.as_ptr()).open_protocol)(
            handle.as_ptr(),
            &mut protocol_guid,
            protocol.as_mut_ptr().cast(),
            system_handle.as_ptr(),
            crate::ptr::null_mut(),
            r_efi::system::OPEN_PROTOCOL_GET_PROTOCOL,
        )
    };

    if r.is_error() {
        Err(status_to_io_error(r))
    } else {
        NonNull::new(unsafe { protocol.assume_init() })
            .ok_or(const_io_error!(io::ErrorKind::Other, "null protocol"))
    }
}

pub(crate) fn status_to_io_error(s: r_efi::efi::Status) -> io::Error {
    use io::ErrorKind;
    use r_efi::efi::Status;

    // Keep the List in Alphabetical Order
    // The Messages are taken from UEFI Specification Appendix D - Status Codes
    match s {
        Status::ABORTED => {
            const_io_error!(ErrorKind::ConnectionAborted, "The operation was aborted.")
        }
        Status::ACCESS_DENIED => {
            const_io_error!(ErrorKind::PermissionDenied, "Access was denied.")
        }
        Status::ALREADY_STARTED => {
            const_io_error!(ErrorKind::Other, "The protocol has already been started.")
        }
        Status::BAD_BUFFER_SIZE => {
            const_io_error!(
                ErrorKind::InvalidData,
                "The buffer was not the proper size for the request."
            )
        }
        Status::BUFFER_TOO_SMALL => {
            const_io_error!(
                ErrorKind::FileTooLarge,
                "The buffer is not large enough to hold the requested data. The required buffer size is returned in the appropriate parameter when this error occurs."
            )
        }
        Status::COMPROMISED_DATA => {
            const_io_error!(
                ErrorKind::Other,
                "The security status of the data is unknown or compromised and the data must be updated or replaced to restore a valid security status."
            )
        }
        Status::CONNECTION_FIN => {
            const_io_error!(
                ErrorKind::Other,
                "The receiving operation fails because the communication peer has closed the connection and there is no more data in the receive buffer of the instance."
            )
        }
        Status::CONNECTION_REFUSED => {
            const_io_error!(
                ErrorKind::ConnectionRefused,
                "The receiving or transmission operation fails because this connection is refused."
            )
        }
        Status::CONNECTION_RESET => {
            const_io_error!(
                ErrorKind::ConnectionReset,
                "The connect fails because the connection is reset either by instance itself or the communication peer."
            )
        }
        Status::CRC_ERROR => const_io_error!(ErrorKind::Other, "A CRC error was detected."),
        Status::DEVICE_ERROR => const_io_error!(
            ErrorKind::Other,
            "The physical device reported an error while attempting the operation."
        ),
        Status::END_OF_FILE => {
            const_io_error!(ErrorKind::UnexpectedEof, "The end of the file was reached.")
        }
        Status::END_OF_MEDIA => {
            const_io_error!(ErrorKind::Other, "Beginning or end of media was reached")
        }
        Status::HOST_UNREACHABLE => {
            const_io_error!(ErrorKind::HostUnreachable, "The remote host is not reachable.")
        }
        Status::HTTP_ERROR => {
            const_io_error!(ErrorKind::Other, "A HTTP error occurred during the network operation.")
        }
        Status::ICMP_ERROR => {
            const_io_error!(
                ErrorKind::Other,
                "An ICMP error occurred during the network operation."
            )
        }
        Status::INCOMPATIBLE_VERSION => {
            const_io_error!(
                ErrorKind::Other,
                "The function encountered an internal version that was incompatible with a version requested by the caller."
            )
        }
        Status::INVALID_LANGUAGE => {
            const_io_error!(ErrorKind::InvalidData, "The language specified was invalid.")
        }
        Status::INVALID_PARAMETER => {
            const_io_error!(ErrorKind::InvalidInput, "A parameter was incorrect.")
        }
        Status::IP_ADDRESS_CONFLICT => {
            const_io_error!(ErrorKind::AddrInUse, "There is an address conflict address allocation")
        }
        Status::LOAD_ERROR => {
            const_io_error!(ErrorKind::Other, "The image failed to load.")
        }
        Status::MEDIA_CHANGED => {
            const_io_error!(
                ErrorKind::Other,
                "The medium in the device has changed since the last access."
            )
        }
        Status::NETWORK_UNREACHABLE => {
            const_io_error!(
                ErrorKind::NetworkUnreachable,
                "The network containing the remote host is not reachable."
            )
        }
        Status::NO_MAPPING => {
            const_io_error!(ErrorKind::Other, "A mapping to a device does not exist.")
        }
        Status::NO_MEDIA => {
            const_io_error!(
                ErrorKind::Other,
                "The device does not contain any medium to perform the operation."
            )
        }
        Status::NO_RESPONSE => {
            const_io_error!(
                ErrorKind::HostUnreachable,
                "The server was not found or did not respond to the request."
            )
        }
        Status::NOT_FOUND => const_io_error!(ErrorKind::NotFound, "The item was not found."),
        Status::NOT_READY => {
            const_io_error!(ErrorKind::ResourceBusy, "There is no data pending upon return.")
        }
        Status::NOT_STARTED => {
            const_io_error!(ErrorKind::Other, "The protocol has not been started.")
        }
        Status::OUT_OF_RESOURCES => {
            const_io_error!(ErrorKind::OutOfMemory, "A resource has run out.")
        }
        Status::PROTOCOL_ERROR => {
            const_io_error!(
                ErrorKind::Other,
                "A protocol error occurred during the network operation."
            )
        }
        Status::PROTOCOL_UNREACHABLE => {
            const_io_error!(ErrorKind::Other, "An ICMP protocol unreachable error is received.")
        }
        Status::SECURITY_VIOLATION => {
            const_io_error!(
                ErrorKind::PermissionDenied,
                "The function was not performed due to a security violation."
            )
        }
        Status::TFTP_ERROR => {
            const_io_error!(ErrorKind::Other, "A TFTP error occurred during the network operation.")
        }
        Status::TIMEOUT => const_io_error!(ErrorKind::TimedOut, "The timeout time expired."),
        Status::UNSUPPORTED => {
            const_io_error!(ErrorKind::Unsupported, "The operation is not supported.")
        }
        Status::VOLUME_FULL => {
            const_io_error!(ErrorKind::StorageFull, "There is no more space on the file system.")
        }
        Status::VOLUME_CORRUPTED => {
            const_io_error!(
                ErrorKind::Other,
                "An inconstancy was detected on the file system causing the operating to fail."
            )
        }
        Status::WRITE_PROTECTED => {
            const_io_error!(ErrorKind::ReadOnlyFilesystem, "The device cannot be written to.")
        }
        _ => io::Error::new(ErrorKind::Uncategorized, format!("Status: {}", s.as_usize())),
    }
}

pub(crate) fn create_event(
    signal: u32,
    tpl: efi::Tpl,
    handler: Option<efi::EventNotify>,
    context: *mut crate::ffi::c_void,
) -> io::Result<NonNull<crate::ffi::c_void>> {
    let boot_services: NonNull<efi::BootServices> =
        boot_services().ok_or(BOOT_SERVICES_UNAVAILABLE)?.cast();
    let mut exit_boot_service_event: r_efi::efi::Event = crate::ptr::null_mut();
    let r = unsafe {
        let create_event = (*boot_services.as_ptr()).create_event;
        (create_event)(signal, tpl, handler, context, &mut exit_boot_service_event)
    };
    if r.is_error() {
        Err(status_to_io_error(r))
    } else {
        NonNull::new(exit_boot_service_event)
            .ok_or(const_io_error!(io::ErrorKind::Other, "null protocol"))
    }
}

/// # SAFETY
/// - The supplied event must be valid
pub(crate) unsafe fn close_event(evt: NonNull<crate::ffi::c_void>) -> io::Result<()> {
    let boot_services: NonNull<efi::BootServices> =
        boot_services().ok_or(BOOT_SERVICES_UNAVAILABLE)?.cast();
    let r = unsafe {
        let close_event = (*boot_services.as_ptr()).close_event;
        (close_event)(evt.as_ptr())
    };

    if r.is_error() { Err(status_to_io_error(r)) } else { Ok(()) }
}
