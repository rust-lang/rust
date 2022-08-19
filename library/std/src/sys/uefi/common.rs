use r_efi::efi::{EventNotify, Guid, Tpl};

use crate::alloc::{AllocError, Allocator, Global, Layout};
use crate::io;
use crate::mem::{self, MaybeUninit};
use crate::ops::{Deref, DerefMut};
use crate::os::uefi;
use crate::os::uefi::io::status_to_io_error;
use crate::ptr::{NonNull, Unique};

pub const BOOT_SERVICES_ERROR: io::Error =
    io::error::const_io_error!(io::ErrorKind::Uncategorized, "Failed to acquire Boot Services",);
pub const RUNTIME_SERVICES_ERROR: io::Error =
    io::error::const_io_error!(io::ErrorKind::Uncategorized, "Failed to acquire Runtime Services",);
pub const SYSTEM_TABLE_ERROR: io::Error =
    io::error::const_io_error!(io::ErrorKind::Uncategorized, "Failed to acquire System Table",);
pub const SYSTEM_HANDLE_ERROR: io::Error =
    io::error::const_io_error!(io::ErrorKind::Uncategorized, "Failed to acquire System Handle",);
pub const VARIABEL_BOX_ERROR: io::Error =
    io::error::const_io_error!(io::ErrorKind::Uncategorized, "Failed to Allocate Varible Box",);

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

// A type to make working with UEFI DSTs easier
// The Layout of this type has to be explicitly supplied
// Inspiered by Box
pub struct VariableBox<T> {
    inner: Unique<T>,
    layout: Layout,
}

impl<T> VariableBox<T> {
    #[inline]
    pub const unsafe fn from_raw(inner: *mut T, layout: Layout) -> VariableBox<T> {
        VariableBox { inner: unsafe { Unique::new_unchecked(inner) }, layout }
    }

    #[inline]
    pub const fn into_raw_with_layout(b: VariableBox<T>) -> (*mut T, Layout) {
        let (leaked, layout) = VariableBox::into_unique(b);
        (leaked.as_ptr(), layout)
    }

    pub fn try_new_uninit(layout: Layout) -> Result<VariableBox<MaybeUninit<T>>, AllocError> {
        let inner = Global.allocate(layout)?.cast();
        unsafe { Ok(VariableBox::from_raw(inner.as_ptr(), layout)) }
    }

    #[inline]
    pub const fn leak<'a>(b: Self) -> &'a mut T {
        unsafe { &mut *mem::ManuallyDrop::new(b).inner.as_ptr() }
    }

    #[inline]
    pub const fn into_unique(b: Self) -> (Unique<T>, Layout) {
        let layout = b.layout;
        (Unique::from(VariableBox::leak(b)), layout)
    }

    #[inline]
    pub fn layout(&self) -> Layout {
        self.layout
    }
}

impl<T> VariableBox<MaybeUninit<T>> {
    #[inline]
    pub unsafe fn assume_init(self) -> VariableBox<T> {
        let (raw, layout) = VariableBox::into_raw_with_layout(self);
        unsafe { VariableBox::from_raw(raw as *mut T, layout) }
    }
}

impl<T> AsRef<T> for VariableBox<T> {
    fn as_ref(&self) -> &T {
        unsafe { self.inner.as_ref() }
    }
}

impl<T> AsMut<T> for VariableBox<T> {
    fn as_mut(&mut self) -> &mut T {
        unsafe { self.inner.as_mut() }
    }
}

impl<T> Deref for VariableBox<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<T> DerefMut for VariableBox<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut()
    }
}

impl<T> Drop for VariableBox<T> {
    fn drop(&mut self) {
        unsafe {
            Global.deallocate(NonNull::new_unchecked(self.inner.as_ptr()).cast(), self.layout)
        }
    }
}

// Locate handles with a particular protocol GUID
/// Implemented using `EFI_BOOT_SERVICES.LocateHandles()`
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

    let boot_services = uefi::env::get_boot_services().ok_or(io::error::const_io_error!(
        io::ErrorKind::Other,
        "Unable to acquire boot services"
    ))?;
    let mut buf_len = 0usize;

    match inner(&mut guid, boot_services, &mut buf_len, crate::ptr::null_mut()) {
        Ok(()) => unreachable!(),
        Err(e) => match e.kind() {
            io::ErrorKind::FileTooLarge => {}
            _ => return Err(e),
        },
    }

    // The returned buf_len is in bytes
    let mut buf: Vec<r_efi::efi::Handle> =
        Vec::with_capacity(buf_len / crate::mem::size_of::<r_efi::efi::Handle>());
    match inner(&mut guid, boot_services, &mut buf_len, buf.as_mut_ptr()) {
        Ok(()) => {
            // SAFETY: This is safe because the call will succeed only if buf_len >= required
            // length. Also, on success, the `buf_len` is updated with the size of bufferv (in
            // bytes) written
            unsafe { buf.set_len(buf_len / crate::mem::size_of::<r_efi::efi::Handle>()) };
            Ok(buf.iter().filter_map(|x| NonNull::new(*x)).collect())
        }
        Err(e) => Err(e),
    }
}

/// Open Protocol on a handle
/// Implemented using `EFI_BOOT_SERVICES.OpenProtocol()`
pub(crate) fn open_protocol<T>(
    handle: NonNull<crate::ffi::c_void>,
    mut protocol_guid: Guid,
) -> io::Result<NonNull<T>> {
    let boot_services = uefi::env::get_boot_services()
        .ok_or(io::error::const_io_error!(io::ErrorKind::Other, "Failed to get BootServices"))?;
    let system_handle = uefi::env::get_system_handle()
        .ok_or(io::error::const_io_error!(io::ErrorKind::Other, "Failed to get System Handle"))?;
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
            .ok_or(io::error::const_io_error!(io::ErrorKind::Other, "Null Protocol"))
    }
}
