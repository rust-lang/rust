use r_efi::efi::{EventNotify, Guid, Tpl};

use crate::io;
use crate::os::uefi;
use crate::os::uefi::io::status_to_io_error;
use crate::ptr::NonNull;

pub const BOOT_SERVICES_ERROR: io::Error =
    io::error::const_io_error!(io::ErrorKind::Uncategorized, "Failed to acquire Boot Services",);
pub const RUNTIME_SERVICES_ERROR: io::Error =
    io::error::const_io_error!(io::ErrorKind::Uncategorized, "Failed to acquire Runtime Services",);
pub const SYSTEM_TABLE_ERROR: io::Error =
    io::error::const_io_error!(io::ErrorKind::Uncategorized, "Failed to acquire System Table",);
pub const SYSTEM_HANDLE_ERROR: io::Error =
    io::error::const_io_error!(io::ErrorKind::Uncategorized, "Failed to acquire System Handle",);

/// Get the Protocol for current system handle.
/// Note: Some protocols need to be manually freed. It is the callers responsibility to do so.
pub(crate) fn get_current_handle_protocol<T>(protocol_guid: &mut Guid) -> Option<NonNull<T>> {
    let system_handle = uefi::env::get_system_handle()?;
    get_handle_protocol(system_handle, protocol_guid)
}

pub(crate) fn get_handle_protocol<T>(
    handle: NonNull<crate::ffi::c_void>,
    protocol_guid: &mut Guid,
) -> Option<NonNull<T>> {
    let boot_services = uefi::env::get_boot_services()?;
    let mut protocol: *mut crate::ffi::c_void = crate::ptr::null_mut();

    let r = unsafe {
        ((*boot_services.as_ptr()).handle_protocol)(handle.as_ptr(), protocol_guid, &mut protocol)
    };

    if r.is_error() { None } else { NonNull::new(protocol.cast()) }
}

#[repr(transparent)]
pub(crate) struct Event {
    inner: NonNull<crate::ffi::c_void>,
}

impl Event {
    fn new(inner: NonNull<crate::ffi::c_void>) -> Self {
        Self { inner }
    }

    fn from_raw_event(ptr: r_efi::efi::Event) -> Option<Self> {
        Some(Self::new(NonNull::new(ptr)?))
    }

    pub(crate) fn create(
        event_type: u32,
        event_tpl: Tpl,
        notify_function: Option<EventNotify>,
        notify_context: Option<NonNull<crate::ffi::c_void>>,
    ) -> io::Result<Self> {
        let boot_services = uefi::env::get_boot_services().ok_or(io::error::const_io_error!(
            io::ErrorKind::Other,
            "Failed to Acquire Boot Services"
        ))?;

        let mut event: r_efi::efi::Event = crate::ptr::null_mut();
        let notify_context = match notify_context {
            None => crate::ptr::null_mut(),
            Some(x) => x.as_ptr(),
        };

        let r = unsafe {
            ((*boot_services.as_ptr()).create_event)(
                event_type,
                event_tpl,
                notify_function,
                notify_context,
                &mut event,
            )
        };

        if r.is_error() {
            Err(status_to_io_error(r))
        } else {
            Self::from_raw_event(event)
                .ok_or(io::error::const_io_error!(io::ErrorKind::Other, "Event is Null"))
        }
    }

    pub(crate) fn wait(&self) -> io::Result<()> {
        let boot_services = uefi::env::get_boot_services().ok_or(io::error::const_io_error!(
            io::ErrorKind::Other,
            "Failed to Acquire Boot Services"
        ))?;

        let mut index = 0usize;
        let r = unsafe {
            ((*boot_services.as_ptr()).wait_for_event)(
                1,
                [self.as_raw_event()].as_mut_ptr(),
                &mut index,
            )
        };

        if r.is_error() { Err(status_to_io_error(r)) } else { Ok(()) }
    }

    pub(crate) fn as_raw_event(&self) -> r_efi::efi::Event {
        self.inner.as_ptr()
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if let Some(boot_services) = uefi::env::get_boot_services() {
            // Always returns EFI_SUCCESS
            let _ = unsafe { ((*boot_services.as_ptr()).close_event)(self.inner.as_ptr()) };
        }
    }
}
