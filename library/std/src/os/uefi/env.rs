//! UEFI-specific extensions to the primitives in `std::env` module

use super::raw::{BootServices, RuntimeServices, SystemTable};
use crate::ffi::c_void;
use crate::io;
use crate::mem::MaybeUninit;
use crate::ptr::NonNull;
use crate::sync::atomic::{AtomicPtr, Ordering};
use r_efi::efi::{Guid, Handle};
use r_efi::system;

static GLOBAL_SYSTEM_TABLE: AtomicPtr<SystemTable> = AtomicPtr::new(crate::ptr::null_mut());
static GLOBAL_SYSTEM_HANDLE: AtomicPtr<c_void> = AtomicPtr::new(crate::ptr::null_mut());

/// Initializes Global Atomic Pointers to SystemTable and Handle.
/// Should only be called once in the program execution under normal circumstances.
/// The caller should ensure that the pointers are valid.
pub(crate) fn init_globals(handle: NonNull<c_void>, system_table: NonNull<SystemTable>) {
    GLOBAL_SYSTEM_TABLE.store(system_table.as_ptr(), Ordering::SeqCst);
    GLOBAL_SYSTEM_HANDLE.store(handle.as_ptr(), Ordering::SeqCst);
}

#[unstable(feature = "uefi_std", issue = "none")]
/// Get the SystemTable Pointer.
pub fn get_system_table() -> Option<NonNull<SystemTable>> {
    NonNull::new(GLOBAL_SYSTEM_TABLE.load(Ordering::SeqCst))
}

#[unstable(feature = "uefi_std", issue = "none")]
/// Get the SystemHandle Pointer.
pub fn get_system_handle() -> Option<NonNull<c_void>> {
    NonNull::new(GLOBAL_SYSTEM_HANDLE.load(Ordering::SeqCst))
}

#[unstable(feature = "uefi_std", issue = "none")]
/// Get the BootServices Pointer.
pub fn get_boot_services() -> Option<NonNull<BootServices>> {
    let system_table = get_system_table()?;
    let boot_services = unsafe { (*system_table.as_ptr()).boot_services };
    NonNull::new(boot_services)
}

#[unstable(feature = "uefi_std", issue = "none")]
/// Get the RuntimeServices Pointer.
pub fn get_runtime_services() -> Option<NonNull<RuntimeServices>> {
    let system_table = get_system_table()?;
    let runtime_services = unsafe { (*system_table.as_ptr()).runtime_services };
    NonNull::new(runtime_services)
}

#[unstable(feature = "uefi_std", issue = "none")]
/// Open Protocol on a handle
/// Implemented using `EFI_BOOT_SERVICES.OpenProtocol()`
pub fn open_protocol<T>(
    handle: NonNull<c_void>,
    mut protocol_guid: Guid,
) -> io::Result<NonNull<T>> {
    let boot_services = get_boot_services()
        .ok_or(io::error::const_io_error!(io::ErrorKind::Other, "Failed to get BootServices"))?;
    let system_handle = get_system_handle()
        .ok_or(io::error::const_io_error!(io::ErrorKind::Other, "Failed to get System Handle"))?;
    let mut protocol: MaybeUninit<*mut T> = MaybeUninit::uninit();

    let r = unsafe {
        ((*boot_services.as_ptr()).open_protocol)(
            handle.as_ptr(),
            &mut protocol_guid,
            protocol.as_mut_ptr().cast(),
            system_handle.as_ptr(),
            crate::ptr::null_mut(),
            system::OPEN_PROTOCOL_GET_PROTOCOL,
        )
    };

    if r.is_error() {
        Err(super::io::status_to_io_error(r))
    } else {
        NonNull::new(unsafe { protocol.assume_init() })
            .ok_or(io::error::const_io_error!(io::ErrorKind::Other, "Null Protocol"))
    }
}

#[unstable(feature = "uefi_std", issue = "none")]
// Locate handles with a particula protocol Guid
/// Implemented using `EFI_BOOT_SERVICES.LocateHandles()`
pub fn locate_handles(mut guid: Guid) -> io::Result<Vec<NonNull<c_void>>> {
    fn inner(
        guid: &mut Guid,
        boot_services: NonNull<BootServices>,
        buf_size: &mut usize,
        buf: *mut Handle,
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

        if r.is_error() { Err(super::io::status_to_io_error(r)) } else { Ok(()) }
    }

    let boot_services = get_boot_services().ok_or(io::error::const_io_error!(
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

    let mut buf: Vec<Handle> = Vec::with_capacity(buf_len);

    match inner(&mut guid, boot_services, &mut buf_len, buf.as_mut_ptr()) {
        Ok(()) => {
            unsafe { buf.set_len(buf_len) };
            Ok(buf.iter().filter_map(|x| NonNull::new(*x)).collect())
        }
        Err(e) => Err(e),
    }
}
