//! UEFI-specific extensions to the primitives in `std::env` module

use super::raw::{BootServices, RuntimeServices, SystemTable};
use crate::ffi::c_void;
use crate::io;
use crate::mem::MaybeUninit;
use crate::ptr::NonNull;
use crate::sync::atomic::{AtomicPtr, Ordering};
use r_efi::efi::{Guid, Handle, Status};
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

/// Get the Protocol for current system handle.
/// Note: Some protocols need to be manually freed. It is the callers responsibility to do so.
pub(crate) fn get_current_handle_protocol<T>(protocol_guid: &mut Guid) -> Option<NonNull<T>> {
    let system_handle = get_system_handle()?;
    get_handle_protocol(system_handle, protocol_guid)
}

pub(crate) fn get_handle_protocol<T>(
    handle: NonNull<c_void>,
    protocol_guid: &mut Guid,
) -> Option<NonNull<T>> {
    let boot_services = get_boot_services()?;
    let mut protocol: *mut c_void = crate::ptr::null_mut();

    let r = unsafe {
        ((*boot_services.as_ptr()).handle_protocol)(handle.as_ptr(), protocol_guid, &mut protocol)
    };

    if r.is_error() { None } else { NonNull::new(protocol.cast()) }
}

pub(crate) fn open_protocol<T>(
    handle: NonNull<c_void>,
    mut protocol_guid: Guid,
) -> io::Result<NonNull<T>> {
    let boot_services = get_boot_services()
        .ok_or(io::Error::new(io::ErrorKind::Other, "Failed to get BootServices"))?;
    let system_handle = get_system_handle()
        .ok_or(io::Error::new(io::ErrorKind::Other, "Failed to get System Handle"))?;
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
        match r {
            Status::INVALID_PARAMETER => {
                Err(io::Error::new(io::ErrorKind::InvalidInput, "EFI_INVALID_PARAMETER"))
            }
            Status::UNSUPPORTED => {
                Err(io::Error::new(io::ErrorKind::Unsupported, "Handle does not support Protocol"))
            }
            Status::ACCESS_DENIED => {
                Err(io::Error::new(io::ErrorKind::PermissionDenied, "EFI_ACCESS_DENIED"))
            }
            Status::ALREADY_STARTED => {
                Err(io::Error::new(io::ErrorKind::Other, "EFI_ALREADY_STARTED"))
            }
            _ => Err(io::Error::new(
                io::ErrorKind::Uncategorized,
                format!("Status: {}", r.as_usize()),
            )),
        }
    } else {
        NonNull::new(unsafe { protocol.assume_init() })
            .ok_or(io::Error::new(io::ErrorKind::Other, "Null Protocol"))
    }
}

pub(crate) fn locate_handles(mut guid: Guid) -> io::Result<Vec<NonNull<c_void>>> {
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

        if r.is_error() {
            match r {
                Status::NOT_FOUND => {
                    Err(io::Error::new(io::ErrorKind::NotFound, "No handles match the search"))
                }
                Status::BUFFER_TOO_SMALL => Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "The BufferSize is too small for the result",
                )),
                Status::INVALID_PARAMETER => {
                    Err(io::Error::new(io::ErrorKind::InvalidInput, "EFI_INVALID_PARAMETER"))
                }
                _ => Err(io::Error::new(
                    io::ErrorKind::Uncategorized,
                    format!("Status: {}", r.as_usize()),
                )),
            }
        } else {
            Ok(())
        }
    }

    let boot_services = get_boot_services()
        .ok_or(io::Error::new(io::ErrorKind::Other, "Unable to acquire boot services"))?;
    let mut buf_len = 0usize;

    match inner(&mut guid, boot_services, &mut buf_len, crate::ptr::null_mut()) {
        Ok(()) => unreachable!(),
        Err(e) => match e.kind() {
            io::ErrorKind::InvalidData => {}
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
