mod implementation;
pub mod tcp4;
pub mod tcp6;

pub use implementation::*;

mod uefi_service_binding {
    use crate::io;
    use crate::mem::MaybeUninit;
    use crate::os::uefi;
    use crate::os::uefi::raw::protocols::service_binding;
    use crate::os::uefi::raw::Status;
    use crate::ptr::NonNull;

    #[derive(Clone, Copy)]
    pub struct ServiceBinding {
        service_binding_guid: uefi::raw::Guid,
        handle: NonNull<crate::ffi::c_void>,
    }

    impl ServiceBinding {
        pub fn new(
            service_binding_guid: uefi::raw::Guid,
            handle: NonNull<crate::ffi::c_void>,
        ) -> Self {
            Self { service_binding_guid, handle }
        }

        pub fn create_child(&self) -> io::Result<NonNull<crate::ffi::c_void>> {
            let service_binding_protocol: NonNull<service_binding::Protocol> =
                uefi::env::open_protocol(self.handle, self.service_binding_guid)?;
            let mut child_handle: MaybeUninit<uefi::raw::Handle> = MaybeUninit::uninit();
            let r = unsafe {
                ((*service_binding_protocol.as_ptr()).create_child)(
                    service_binding_protocol.as_ptr(),
                    child_handle.as_mut_ptr(),
                )
            };

            if r.is_error() {
                match r {
                    Status::INVALID_PARAMETER => {
                        Err(io::Error::new(io::ErrorKind::InvalidInput, "ChildHandle is NULL"))
                    }
                    Status::OUT_OF_RESOURCES => Err(io::Error::new(
                        io::ErrorKind::OutOfMemory,
                        "There are not enough resources available to create the child",
                    )),
                    _ => Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!("Unknown Error: {}", r.as_usize()),
                    )),
                }
            } else {
                NonNull::new(unsafe { child_handle.assume_init() })
                    .ok_or(io::Error::new(io::ErrorKind::Other, "Null Handle"))
            }
        }

        pub fn destroy_child(&self, child_handle: NonNull<crate::ffi::c_void>) -> io::Result<()> {
            let service_binding_protocol: NonNull<service_binding::Protocol> =
                uefi::env::open_protocol(self.handle, self.service_binding_guid)?;
            let r = unsafe {
                ((*service_binding_protocol.as_ptr()).destroy_child)(
                    service_binding_protocol.as_ptr(),
                    child_handle.as_ptr(),
                )
            };

            if r.is_error() {
                match r {
                    Status::UNSUPPORTED => Err(io::Error::new(
                        io::ErrorKind::Unsupported,
                        "ChildHandle does not support the protocol that is being removed",
                    )),
                    Status::INVALID_PARAMETER => Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "ChildHandle is not a valid UEFI handle",
                    )),
                    Status::ACCESS_DENIED => Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "The protocol could not be removed from the ChildHandle because its services are being used",
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
    }
}
