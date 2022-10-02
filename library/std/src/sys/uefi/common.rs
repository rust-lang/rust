use r_efi::efi::{EventNotify, Guid, Tpl};

use crate::alloc::{AllocError, Allocator, Global, Layout};
use crate::ffi::{OsStr, OsString};
use crate::io::{self, const_io_error};
use crate::mem::{self, MaybeUninit};
use crate::os::uefi;
use crate::os::uefi::ffi::{OsStrExt, OsStringExt};
use crate::ptr::NonNull;

/// Get the Protocol for current system handle.
/// Note: Some protocols need to be manually freed. It is the callers responsibility to do so.
pub(crate) fn get_current_handle_protocol<T>(protocol_guid: Guid) -> Option<NonNull<T>> {
    let system_handle = uefi::env::image_handle();
    open_protocol(system_handle, protocol_guid).ok()
}

#[repr(transparent)]
pub(crate) struct Event {
    inner: NonNull<crate::ffi::c_void>,
}

impl Event {
    #[inline]
    fn new(inner: NonNull<crate::ffi::c_void>) -> Self {
        Self { inner }
    }

    #[inline]
    fn from_raw_event(ptr: r_efi::efi::Event) -> Option<Self> {
        Some(Self::new(NonNull::new(ptr)?))
    }

    pub(crate) fn create(
        event_type: u32,
        event_tpl: Tpl,
        notify_function: Option<EventNotify>,
        notify_context: Option<NonNull<crate::ffi::c_void>>,
    ) -> io::Result<Self> {
        let boot_services = boot_services();

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
                .ok_or(const_io_error!(io::ErrorKind::Other, "event is null"))
        }
    }

    pub(crate) fn wait(&self) -> io::Result<()> {
        let boot_services = boot_services();

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

    #[inline]
    pub(crate) fn as_raw_event(&self) -> r_efi::efi::Event {
        self.inner.as_ptr()
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        let boot_services = boot_services();
        // Always returns EFI_SUCCESS
        let _ = unsafe { ((*boot_services.as_ptr()).close_event)(self.inner.as_ptr()) };
    }
}

// A type to make working with UEFI DSTs easier
// The Layout of this type has to be explicitly supplied
// Inspiered by Box
pub(crate) struct VariableBox<T> {
    inner: NonNull<T>,
    layout: Layout,
}

impl<T> VariableBox<T> {
    pub(crate) unsafe fn from_raw(inner: *mut T, layout: Layout) -> VariableBox<T> {
        unsafe { VariableBox::new(NonNull::new_unchecked(inner), layout) }
    }

    #[inline]
    pub(crate) unsafe fn new(inner: NonNull<T>, layout: Layout) -> VariableBox<T> {
        VariableBox { inner, layout }
    }

    #[inline]
    pub(crate) fn into_raw_with_layout(b: VariableBox<T>) -> (*mut T, Layout) {
        let (leaked, layout) = VariableBox::into_non_null(b);
        (leaked.as_ptr(), layout)
    }

    pub(crate) fn new_uninit(layout: Layout) -> VariableBox<MaybeUninit<T>> {
        match Self::try_new_uninit(layout) {
            Ok(x) => x,
            Err(_) => crate::alloc::handle_alloc_error(layout),
        }
    }

    fn try_new_uninit(layout: Layout) -> Result<VariableBox<MaybeUninit<T>>, AllocError> {
        let inner = Global.allocate(layout)?.cast();
        unsafe { Ok(VariableBox::new(inner, layout)) }
    }

    #[inline]
    pub(crate) fn leak<'a>(b: Self) -> &'a mut T {
        unsafe { &mut *mem::ManuallyDrop::new(b).inner.as_ptr() }
    }

    #[inline]
    pub(crate) fn into_non_null(b: Self) -> (NonNull<T>, Layout) {
        let layout = b.layout;
        (NonNull::from(VariableBox::leak(b)), layout)
    }

    #[inline]
    pub(crate) fn layout(&self) -> Layout {
        self.layout
    }

    #[inline]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.inner.as_ptr()
    }

    #[inline]
    pub(crate) fn as_ptr(&self) -> *const T {
        self.inner.as_ptr()
    }
}

impl<T> VariableBox<MaybeUninit<T>> {
    #[inline]
    pub(crate) unsafe fn assume_init(self) -> VariableBox<T> {
        let (raw, layout) = VariableBox::into_raw_with_layout(self);
        unsafe { VariableBox::from_raw(raw.cast(), layout) }
    }

    #[inline]
    pub(crate) fn as_uninit_mut_ptr(&mut self) -> *mut T {
        unsafe { (*self.inner.as_ptr()).as_mut_ptr() }
    }
}

impl<T> Drop for VariableBox<T> {
    #[inline]
    fn drop(&mut self) {
        unsafe { Global.deallocate(self.inner.cast(), self.layout) }
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

    let boot_services = boot_services();
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
    let boot_services = boot_services();
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
    match s {
        Status::ABORTED => {
            const_io_error!(ErrorKind::ConnectionAborted, "EFI_ABORTED")
        }
        Status::ACCESS_DENIED => {
            const_io_error!(ErrorKind::PermissionDenied, "EFI_ACCESS_DENIED")
        }
        Status::ALREADY_STARTED => {
            const_io_error!(ErrorKind::Other, "EFI_ALREADY_STARTED")
        }
        Status::BAD_BUFFER_SIZE => {
            const_io_error!(ErrorKind::InvalidData, "EFI_BAD_BUFFER_SIZE")
        }
        Status::BUFFER_TOO_SMALL => {
            const_io_error!(ErrorKind::FileTooLarge, "EFI_BUFFER_TOO_SMALL")
        }
        Status::COMPROMISED_DATA => {
            const_io_error!(ErrorKind::Other, "EFI_COMPRIMISED_DATA")
        }
        Status::CONNECTION_FIN => {
            const_io_error!(ErrorKind::ConnectionAborted, "EFI_CONNECTION_FIN")
        }
        Status::CONNECTION_REFUSED => {
            const_io_error!(ErrorKind::ConnectionRefused, "EFI_CONNECTION_REFUSED")
        }
        Status::CONNECTION_RESET => {
            const_io_error!(ErrorKind::ConnectionReset, "EFI_CONNECTION_RESET")
        }
        Status::CRC_ERROR => const_io_error!(ErrorKind::InvalidData, "EFI_CRC_ERROR"),
        Status::DEVICE_ERROR => const_io_error!(ErrorKind::Other, "EFI_DEVICE_ERROR"),
        Status::END_OF_FILE => {
            const_io_error!(ErrorKind::UnexpectedEof, "EFI_END_OF_FILE")
        }
        Status::END_OF_MEDIA => {
            const_io_error!(ErrorKind::UnexpectedEof, "EFI_END_OF_MEDIA")
        }
        Status::HOST_UNREACHABLE => {
            const_io_error!(ErrorKind::HostUnreachable, "EFI_HOST_UNREACHABLE")
        }
        Status::HTTP_ERROR => {
            const_io_error!(ErrorKind::NetworkUnreachable, "EFI_HTTP_ERROR")
        }
        Status::ICMP_ERROR => {
            const_io_error!(ErrorKind::Other, "EFI_ICMP_ERROR")
        }
        Status::INCOMPATIBLE_VERSION => {
            const_io_error!(ErrorKind::Other, "EFI_INCOMPATIBLE_VERSION")
        }
        Status::INVALID_LANGUAGE => {
            const_io_error!(ErrorKind::InvalidData, "EFI_INVALID_LANGUAGE")
        }
        Status::INVALID_PARAMETER => {
            const_io_error!(ErrorKind::InvalidInput, "EFI_INVALID_PARAMETER")
        }
        Status::IP_ADDRESS_CONFLICT => {
            const_io_error!(ErrorKind::AddrInUse, "EFI_IP_ADDRESS_CONFLICT")
        }
        Status::LOAD_ERROR => {
            const_io_error!(ErrorKind::Other, "EFI_LOAD_ERROR")
        }
        Status::MEDIA_CHANGED => {
            const_io_error!(ErrorKind::StaleNetworkFileHandle, "EFI_MEDIA_CHANGED")
        }
        Status::NETWORK_UNREACHABLE => {
            const_io_error!(ErrorKind::NetworkUnreachable, "EFI_NETWORK_UNREACHABLE")
        }
        Status::NO_MAPPING => {
            const_io_error!(ErrorKind::Other, "EFI_NO_MAPPING")
        }
        Status::NO_MEDIA => {
            const_io_error!(ErrorKind::Other, "EFI_NO_MEDIA")
        }
        Status::NO_RESPONSE => {
            const_io_error!(ErrorKind::HostUnreachable, "EFI_NO_RESPONSE")
        }
        Status::NOT_FOUND => const_io_error!(ErrorKind::NotFound, "EFI_NOT_FOUND"),
        Status::NOT_READY => const_io_error!(ErrorKind::ResourceBusy, "EFI_NOT_READY"),
        Status::NOT_STARTED => const_io_error!(ErrorKind::Other, "EFI_NOT_STARTED"),
        Status::OUT_OF_RESOURCES => {
            const_io_error!(ErrorKind::OutOfMemory, "EFI_OUT_OF_RESOURCES")
        }
        Status::PROTOCOL_ERROR => {
            const_io_error!(ErrorKind::Other, "EFI_PROTOCOL_ERROR")
        }
        Status::PROTOCOL_UNREACHABLE => {
            const_io_error!(ErrorKind::Other, "EFI_PROTOCOL_UNREACHABLE")
        }
        Status::SECURITY_VIOLATION => {
            const_io_error!(ErrorKind::PermissionDenied, "EFI_SECURITY_VIOLATION")
        }
        Status::TFTP_ERROR => const_io_error!(ErrorKind::Other, "EFI_TFTP_ERROR"),
        Status::TIMEOUT => const_io_error!(ErrorKind::TimedOut, "EFI_TIMEOUT"),
        Status::UNSUPPORTED => {
            const_io_error!(ErrorKind::Unsupported, "EFI_UNSUPPORTED")
        }
        Status::VOLUME_FULL => {
            const_io_error!(ErrorKind::StorageFull, "EFI_VOLUME_FULL")
        }
        Status::VOLUME_CORRUPTED => {
            const_io_error!(ErrorKind::Other, "EFI_VOLUME_CORRUPTED")
        }
        Status::WRITE_PROTECTED => {
            const_io_error!(ErrorKind::ReadOnlyFilesystem, "EFI_WRITE_PROTECTED")
        }
        _ => io::Error::new(ErrorKind::Uncategorized, format!("Status: {}", s.as_usize())),
    }
}

pub(crate) fn install_protocol<T>(
    handle: &mut r_efi::efi::Handle,
    mut guid: r_efi::efi::Guid,
    interface: &mut T,
) -> io::Result<()> {
    let boot_services = boot_services();
    let r = unsafe {
        ((*boot_services.as_ptr()).install_protocol_interface)(
            handle,
            &mut guid,
            r_efi::efi::NATIVE_INTERFACE,
            interface as *mut T as *mut crate::ffi::c_void,
        )
    };
    if r.is_error() { Err(status_to_io_error(r)) } else { Ok(()) }
}

pub(crate) fn uninstall_protocol<T>(
    handle: r_efi::efi::Handle,
    mut guid: r_efi::efi::Guid,
    interface: &mut T,
) -> io::Result<()> {
    let boot_services = boot_services();
    let r = unsafe {
        ((*boot_services.as_ptr()).uninstall_protocol_interface)(
            handle,
            &mut guid,
            interface as *mut T as *mut crate::ffi::c_void,
        )
    };
    if r.is_error() { Err(status_to_io_error(r)) } else { Ok(()) }
}

// A Helper trait for `ProtocolWrapper<P>`.
pub(crate) trait Protocol {
    const PROTOCOL_GUID: Guid;
}

// A wrapper for creating protocols from Rust in Local scope. Uninstalls the protocol on Drop.
pub(crate) struct ProtocolWrapper<P>
where
    P: Protocol,
{
    handle: NonNull<crate::ffi::c_void>,
    protocol: Box<P>,
}

impl<P> ProtocolWrapper<P>
where
    P: Protocol,
{
    #[inline]
    pub(crate) fn new(handle: NonNull<crate::ffi::c_void>, protocol: Box<P>) -> Self {
        Self { handle, protocol }
    }

    pub(crate) fn install_protocol(protocol: P) -> io::Result<ProtocolWrapper<P>> {
        let mut handle: r_efi::efi::Handle = crate::ptr::null_mut();
        let mut protocol = Box::new(protocol);
        install_protocol::<P>(&mut handle, P::PROTOCOL_GUID, &mut protocol)?;
        let handle = NonNull::new(handle)
            .ok_or(io::const_io_error!(io::ErrorKind::Uncategorized, "found null handle"))?;
        Ok(Self::new(handle, protocol))
    }

    pub(crate) fn install_protocol_in(
        protocol: P,
        mut handle: r_efi::efi::Handle,
    ) -> io::Result<ProtocolWrapper<P>> {
        let mut protocol = Box::new(protocol);
        install_protocol::<P>(&mut handle, P::PROTOCOL_GUID, &mut protocol)?;
        let handle = NonNull::new(handle)
            .ok_or(io::const_io_error!(io::ErrorKind::Uncategorized, "found null handle"))?;
        Ok(Self::new(handle, protocol))
    }

    #[inline]
    pub(crate) fn handle(&self) -> NonNull<crate::ffi::c_void> {
        self.handle
    }
}

impl<P> Drop for ProtocolWrapper<P>
where
    P: Protocol,
{
    #[inline]
    fn drop(&mut self) {
        let _ = uninstall_protocol::<P>(self.handle.as_ptr(), P::PROTOCOL_GUID, &mut self.protocol);
    }
}

/// Get the BootServices Pointer.
pub(crate) fn boot_services() -> NonNull<r_efi::efi::BootServices> {
    let system_table: NonNull<r_efi::efi::SystemTable> = uefi::env::system_table().cast();
    let boot_services = unsafe { (*system_table.as_ptr()).boot_services };
    NonNull::new(boot_services).unwrap()
}
/// Get the BootServices Pointer.
/// This function is mostly intended for places where panic is not an option
pub(crate) fn try_boot_services() -> Option<NonNull<r_efi::efi::BootServices>> {
    let system_table: NonNull<r_efi::efi::SystemTable> = uefi::env::try_system_table()?.cast();
    let boot_services = unsafe { (*system_table.as_ptr()).boot_services };
    NonNull::new(boot_services)
}

/// Get the RuntimeServices Pointer.
pub(crate) fn runtime_services() -> NonNull<r_efi::efi::RuntimeServices> {
    let system_table: NonNull<r_efi::efi::SystemTable> = uefi::env::system_table().cast();
    let runtime_services = unsafe { (*system_table.as_ptr()).runtime_services };
    NonNull::new(runtime_services).unwrap()
}

// Create UCS-2 Vector from OsStr
pub(crate) fn to_ffi_string(s: &OsStr) -> Vec<u16> {
    let mut v: Vec<u16> = s.encode_wide().collect();
    v.push(0);
    v
}

// Create OsString from UEFI UCS-2 String
pub(crate) fn from_ffi_string(ucs: *mut u16, bytes: usize) -> OsString {
    // Convert len in bytes to string length and do not count the null character
    let len = bytes / crate::mem::size_of::<u16>() - 1;
    let s = unsafe { crate::slice::from_raw_parts(ucs, len) };
    OsString::from_wide(s)
}
