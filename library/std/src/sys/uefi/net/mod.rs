mod implementation;
mod tcp;
mod tcp4;
mod tcp6;

pub use implementation::*;

mod uefi_service_binding {
    use super::super::common::{self, status_to_io_error};
    use crate::io;
    use crate::mem::MaybeUninit;
    use crate::ptr::NonNull;
    use r_efi::protocols::service_binding;

    #[derive(Clone, Copy)]
    pub struct ServiceBinding {
        service_binding_guid: r_efi::efi::Guid,
        handle: NonNull<crate::ffi::c_void>,
    }

    impl ServiceBinding {
        pub fn new(
            service_binding_guid: r_efi::efi::Guid,
            handle: NonNull<crate::ffi::c_void>,
        ) -> Self {
            Self { service_binding_guid, handle }
        }

        pub fn create_child(&self) -> io::Result<NonNull<crate::ffi::c_void>> {
            let service_binding_protocol: NonNull<service_binding::Protocol> =
                common::open_protocol(self.handle, self.service_binding_guid)?;
            let mut child_handle: MaybeUninit<r_efi::efi::Handle> = MaybeUninit::uninit();
            let r = unsafe {
                ((*service_binding_protocol.as_ptr()).create_child)(
                    service_binding_protocol.as_ptr(),
                    child_handle.as_mut_ptr(),
                )
            };

            if r.is_error() {
                Err(status_to_io_error(r))
            } else {
                NonNull::new(unsafe { child_handle.assume_init() })
                    .ok_or(io::error::const_io_error!(io::ErrorKind::Other, "Null Handle"))
            }
        }

        pub fn destroy_child(&self, child_handle: NonNull<crate::ffi::c_void>) -> io::Result<()> {
            let service_binding_protocol: NonNull<service_binding::Protocol> =
                common::open_protocol(self.handle, self.service_binding_guid)?;
            let r = unsafe {
                ((*service_binding_protocol.as_ptr()).destroy_child)(
                    service_binding_protocol.as_ptr(),
                    child_handle.as_ptr(),
                )
            };

            if r.is_error() { Err(status_to_io_error(r)) } else { Ok(()) }
        }
    }
}

#[inline]
pub(crate) fn ipv4_to_r_efi(ip: &crate::net::Ipv4Addr) -> r_efi::efi::Ipv4Address {
    r_efi::efi::Ipv4Address { addr: ip.octets() }
}

#[inline]
pub(crate) fn ipv4_from_r_efi(ip: r_efi::efi::Ipv4Address) -> crate::net::Ipv4Addr {
    crate::net::Ipv4Addr::from(ip.addr)
}
