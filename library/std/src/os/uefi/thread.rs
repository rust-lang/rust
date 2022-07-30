use crate::ffi::c_void;
use crate::io;
use crate::ptr::NonNull;
use r_efi::efi::{EventNotify, Status, Tpl};

#[repr(transparent)]
pub(crate) struct Event {
    inner: NonNull<c_void>,
}

impl Event {
    fn new(inner: NonNull<c_void>) -> Self {
        Self { inner }
    }

    fn from_raw_event(ptr: r_efi::efi::Event) -> Option<Self> {
        Some(Self::new(NonNull::new(ptr)?))
    }

    pub(crate) fn create(
        event_type: u32,
        event_tpl: Tpl,
        notify_function: Option<EventNotify>,
        notify_context: Option<NonNull<c_void>>,
    ) -> io::Result<Self> {
        let boot_services = super::env::get_boot_services()
            .ok_or(io::Error::new(io::ErrorKind::Other, "Failed to Acquire Boot Services"))?;

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
            match r {
                Status::INVALID_PARAMETER => {
                    Err(io::Error::new(io::ErrorKind::InvalidInput, "EFI_INVALID_PARAMETER"))
                }
                Status::OUT_OF_RESOURCES => Err(io::Error::new(
                    io::ErrorKind::OutOfMemory,
                    "The event could not be allocated",
                )),
                _ => Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("Unknown Error: {}", r.as_usize()),
                )),
            }
        } else {
            Self::from_raw_event(event).ok_or(io::Error::new(io::ErrorKind::Other, "Event is Null"))
        }
    }

    pub(crate) fn wait(&self) -> io::Result<()> {
        let boot_services = super::env::get_boot_services()
            .ok_or(io::Error::new(io::ErrorKind::Other, "Failed to Acquire Boot Services"))?;

        let mut index = 0usize;
        let r = unsafe {
            ((*boot_services.as_ptr()).wait_for_event)(
                1,
                [self.as_raw_event()].as_mut_ptr(),
                &mut index,
            )
        };

        if r.is_error() {
            match r {
                Status::INVALID_PARAMETER => {
                    Err(io::Error::new(io::ErrorKind::InvalidInput, "EFI_INVALID_PARAMETER"))
                }
                Status::UNSUPPORTED => Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "The current TPL is not TPL_APPLICATION.",
                )),
                _ => Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("Unknown Error: {}", r.as_usize()),
                )),
            }
        } else {
            Ok(())
        }
    }

    pub(crate) fn as_raw_event(&self) -> r_efi::efi::Event {
        self.inner.as_ptr()
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if let Some(boot_services) = super::env::get_boot_services() {
            // Always returns EFI_SUCCESS
            let _ = unsafe { ((*boot_services.as_ptr()).close_event)(self.inner.as_ptr()) };
        }
    }
}
