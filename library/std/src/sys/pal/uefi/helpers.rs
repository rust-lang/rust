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
use r_efi::protocols::{device_path, device_path_to_text};

use crate::ffi::OsString;
use crate::io::{self, const_io_error};
use crate::mem::{size_of, MaybeUninit};
use crate::os::uefi::{self, env::boot_services, ffi::OsStringExt};
use crate::ptr::NonNull;
use crate::slice;
use crate::sync::atomic::{AtomicPtr, Ordering};
use crate::sys_common::wstr::WStrUnits;

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

        if r.is_error() { Err(crate::io::Error::from_raw_os_error(r.as_usize())) } else { Ok(()) }
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
    assert_eq!(buf_len % size_of::<r_efi::efi::Handle>(), 0);
    let num_of_handles = buf_len / size_of::<r_efi::efi::Handle>();
    let mut buf: Vec<r_efi::efi::Handle> = Vec::with_capacity(num_of_handles);
    match inner(&mut guid, boot_services, &mut buf_len, buf.as_mut_ptr()) {
        Ok(()) => {
            // This is safe because the call will succeed only if buf_len >= required length.
            // Also, on success, the `buf_len` is updated with the size of bufferv (in bytes) written
            unsafe { buf.set_len(num_of_handles) };
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
        Err(crate::io::Error::from_raw_os_error(r.as_usize()))
    } else {
        NonNull::new(unsafe { protocol.assume_init() })
            .ok_or(const_io_error!(io::ErrorKind::Other, "null protocol"))
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
    let mut event: r_efi::efi::Event = crate::ptr::null_mut();
    let r = unsafe {
        let create_event = (*boot_services.as_ptr()).create_event;
        (create_event)(signal, tpl, handler, context, &mut event)
    };
    if r.is_error() {
        Err(crate::io::Error::from_raw_os_error(r.as_usize()))
    } else {
        NonNull::new(event).ok_or(const_io_error!(io::ErrorKind::Other, "null protocol"))
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

    if r.is_error() { Err(crate::io::Error::from_raw_os_error(r.as_usize())) } else { Ok(()) }
}

/// Get the Protocol for current system handle.
/// Note: Some protocols need to be manually freed. It is the callers responsibility to do so.
pub(crate) fn image_handle_protocol<T>(protocol_guid: Guid) -> io::Result<NonNull<T>> {
    let system_handle = uefi::env::try_image_handle().ok_or(io::const_io_error!(
        io::ErrorKind::NotFound,
        "Protocol not found in Image handle"
    ))?;
    open_protocol(system_handle, protocol_guid)
}

pub(crate) fn device_path_to_text(path: NonNull<device_path::Protocol>) -> io::Result<OsString> {
    fn path_to_text(
        protocol: NonNull<device_path_to_text::Protocol>,
        path: NonNull<device_path::Protocol>,
    ) -> io::Result<OsString> {
        let path_ptr: *mut r_efi::efi::Char16 = unsafe {
            ((*protocol.as_ptr()).convert_device_path_to_text)(
                path.as_ptr(),
                // DisplayOnly
                r_efi::efi::Boolean::FALSE,
                // AllowShortcuts
                r_efi::efi::Boolean::FALSE,
            )
        };

        // SAFETY: `convert_device_path_to_text` returns a pointer to a null-terminated UTF-16
        // string, and that string cannot be deallocated prior to dropping the `WStrUnits`, so
        // it's safe for `WStrUnits` to use.
        let path_len = unsafe {
            WStrUnits::new(path_ptr)
                .ok_or(io::const_io_error!(io::ErrorKind::InvalidData, "Invalid path"))?
                .count()
        };

        let path = OsString::from_wide(unsafe { slice::from_raw_parts(path_ptr.cast(), path_len) });

        if let Some(boot_services) = crate::os::uefi::env::boot_services() {
            let boot_services: NonNull<r_efi::efi::BootServices> = boot_services.cast();
            unsafe {
                ((*boot_services.as_ptr()).free_pool)(path_ptr.cast());
            }
        }

        Ok(path)
    }

    static LAST_VALID_HANDLE: AtomicPtr<crate::ffi::c_void> =
        AtomicPtr::new(crate::ptr::null_mut());

    if let Some(handle) = NonNull::new(LAST_VALID_HANDLE.load(Ordering::Acquire)) {
        if let Ok(protocol) = open_protocol::<device_path_to_text::Protocol>(
            handle,
            device_path_to_text::PROTOCOL_GUID,
        ) {
            return path_to_text(protocol, path);
        }
    }

    let device_path_to_text_handles = locate_handles(device_path_to_text::PROTOCOL_GUID)?;
    for handle in device_path_to_text_handles {
        if let Ok(protocol) = open_protocol::<device_path_to_text::Protocol>(
            handle,
            device_path_to_text::PROTOCOL_GUID,
        ) {
            LAST_VALID_HANDLE.store(handle.as_ptr(), Ordering::Release);
            return path_to_text(protocol, path);
        }
    }

    Err(io::const_io_error!(io::ErrorKind::NotFound, "No device path to text protocol found"))
}
