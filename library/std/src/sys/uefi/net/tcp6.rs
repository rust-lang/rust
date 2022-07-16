use super::uefi_service_binding::ServiceBinding;
use crate::io;
use crate::mem::MaybeUninit;
use crate::net::{Ipv6Addr, SocketAddrV6};
use crate::os::uefi;
use crate::os::uefi::raw::protocols::{
    ip6, managed_network, service_binding, simple_network, tcp6,
};
use crate::os::uefi::raw::Status;
use crate::ptr::NonNull;
use crate::sys_common::AsInner;

// FIXME: Discuss what the values these constants should have
const TRAFFIC_CLASS: u8 = 0;
const HOP_LIMIT: u8 = 255;

pub struct Tcp6Protocol {
    protocol: NonNull<tcp6::Protocol>,
    service_binding: ServiceBinding,
    child_handle: NonNull<crate::ffi::c_void>,
}

impl Tcp6Protocol {
    fn new(
        protocol: NonNull<tcp6::Protocol>,
        service_binding: ServiceBinding,
        child_handle: NonNull<crate::ffi::c_void>,
    ) -> Self {
        Self { protocol, service_binding, child_handle }
    }

    fn with_child_handle(
        service_binding: ServiceBinding,
        child_handle: NonNull<crate::ffi::c_void>,
    ) -> io::Result<Self> {
        let tcp6_protocol = uefi::env::open_protocol(child_handle, tcp6::PROTOCOL_GUID)?;
        Ok(Self::new(tcp6_protocol, service_binding, child_handle))
    }

    fn get_config_data(&self) -> io::Result<tcp6::ConfigData> {
        let protocol = self.protocol.as_ptr();

        let mut state: MaybeUninit<tcp6::ConnectionState> = MaybeUninit::uninit();
        let mut config_data: MaybeUninit<tcp6::ConfigData> = MaybeUninit::uninit();
        let mut ip6_mode_data: MaybeUninit<ip6::ModeData> = MaybeUninit::uninit();
        let mut mnp_mode_data: MaybeUninit<managed_network::ConfigData> = MaybeUninit::uninit();
        let mut snp_mode_data: MaybeUninit<simple_network::Mode> = MaybeUninit::uninit();

        let r = unsafe {
            ((*protocol).get_mode_data)(
                protocol,
                state.as_mut_ptr(),
                config_data.as_mut_ptr(),
                ip6_mode_data.as_mut_ptr(),
                mnp_mode_data.as_mut_ptr(),
                snp_mode_data.as_mut_ptr(),
            )
        };

        if r.is_error() {
            match r {
                Status::NOT_STARTED => Err(io::Error::new(
                    io::ErrorKind::Other,
                    "No configuration data is available because this instance hasn’t been started",
                )),
                Status::INVALID_PARAMETER => {
                    Err(io::Error::new(io::ErrorKind::InvalidInput, "This is NULL"))
                }
                _ => Err(io::Error::new(
                    io::ErrorKind::Uncategorized,
                    format!("Status: {}", r.as_usize()),
                )),
            }
        } else {
            Ok(unsafe { config_data.assume_init() })
        }
    }

    pub fn create(service_binding: ServiceBinding) -> io::Result<Tcp6Protocol> {
        let child_handle = service_binding.create_child()?;
        Self::with_child_handle(service_binding, child_handle)
    }

    pub fn config(
        &self,
        active_flag: bool,
        station_addr: &crate::net::SocketAddrV6,
        remote_addr: &crate::net::SocketAddrV6,
    ) -> io::Result<()> {
        let protocol = self.protocol.as_ptr();

        let mut config_data = tcp6::ConfigData {
            // FIXME: Check in mailing list what traffic_class should be used
            traffic_class: TRAFFIC_CLASS,
            // FIXME: Check in mailing list what hop_limit should be used
            hop_limit: HOP_LIMIT,
            access_point: tcp6::AccessPoint {
                station_address: uefi::raw::Ipv6Address::from(station_addr.ip()),
                station_port: station_addr.port(),
                remote_address: uefi::raw::Ipv6Address::from(remote_addr.ip()),
                remote_port: remote_addr.port(),
                active_flag: uefi::raw::Boolean::from(active_flag),
            },
            // FIXME: Maybe provide a rust default one at some point
            control_option: crate::ptr::null_mut(),
        };

        let r = unsafe { ((*protocol).configure)(protocol, &mut config_data) };

        if r.is_error() {
            let e = match r {
                Status::NO_MAPPING => io::Error::new(
                    io::ErrorKind::Other,
                    "The underlying IPv6 driver was responsible for choosing a source address for this instance, but no source address was available for use",
                ),
                Status::INVALID_PARAMETER => {
                    io::Error::new(io::ErrorKind::InvalidInput, "EFI_INVALID_PARAMETER")
                }
                Status::ACCESS_DENIED => io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "Configuring TCP instance when it is configured without calling Configure() with NULL to reset it",
                ),
                Status::UNSUPPORTED => io::Error::new(
                    io::ErrorKind::Unsupported,
                    "One or more of the control options are not supported in the implementation.",
                ),
                Status::OUT_OF_RESOURCES => io::Error::new(
                    io::ErrorKind::OutOfMemory,
                    "Could not allocate enough system resources when executing Configure()",
                ),
                Status::DEVICE_ERROR => io::Error::new(
                    io::ErrorKind::Other,
                    "An unexpected network or system error occurred",
                ),
                _ => {
                    io::Error::new(io::ErrorKind::Other, format!("Unknown Error: {}", r.as_usize()))
                }
            };
            Err(e)
        } else {
            Ok(())
        }
    }

    pub fn accept(&self) -> io::Result<Tcp6Protocol> {
        let protocol = self.protocol.as_ptr();

        let accept_event = uefi::thread::Event::create(
            uefi::raw::EVT_NOTIFY_SIGNAL,
            uefi::raw::TPL_CALLBACK,
            Some(nop_notify),
            None,
        )?;
        let completion_token =
            tcp6::CompletionToken { event: accept_event.as_raw_event(), status: Status::ABORTED };

        let mut listen_token = tcp6::ListenToken {
            completion_token,
            new_child_handle: unsafe { MaybeUninit::<uefi::raw::Handle>::uninit().assume_init() },
        };

        let r = unsafe { ((*protocol).accept)(protocol, &mut listen_token) };

        if r.is_error() {
            return match r {
                Status::NOT_STARTED => Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "This EFI TCPv6 Protocol instance has not been configured",
                )),
                Status::ACCESS_DENIED => {
                    Err(io::Error::new(io::ErrorKind::PermissionDenied, "EFI_ACCESS_DENIED"))
                }
                Status::INVALID_PARAMETER => {
                    Err(io::Error::new(io::ErrorKind::InvalidInput, "EFI_INVALID_PARAMETER"))
                }
                Status::OUT_OF_RESOURCES => Err(io::Error::new(
                    io::ErrorKind::OutOfMemory,
                    "Could not allocate enough resource to finish the operation",
                )),
                _ => Err(io::Error::new(
                    io::ErrorKind::Uncategorized,
                    format!("Status: {}", r.as_usize()),
                )),
            };
        }
        println!("Wait");
        // accept_event.wait()?;
        // Seems like a bad idea
        while listen_token.completion_token.status == Status::ABORTED {}
        println!("Wait Done");

        let r = listen_token.completion_token.status;
        if r.is_error() {
            match r {
                Status::CONNECTION_RESET => Err(io::Error::new(
                    io::ErrorKind::ConnectionReset,
                    "The accept fails because the
connection is reset either by instance itself or communication peer",
                )),
                Status::ABORTED => Err(io::Error::new(
                    io::ErrorKind::ConnectionAborted,
                    "The accept request has been aborted",
                )),
                Status::SECURITY_VIOLATION => Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "The accept operation was failed because of IPSec policy check",
                )),
                _ => Err(io::Error::new(io::ErrorKind::Other, format!("Status: {}", r.as_usize()))),
            }
        } else {
            let child_handle = NonNull::new(listen_token.new_child_handle)
                .ok_or(io::Error::new(io::ErrorKind::Other, "Null Child Handle"))?;
            Self::with_child_handle(self.service_binding, child_handle)
        }
    }

    pub fn connect(&self) -> io::Result<()> {
        todo!()
    }

    pub fn transmit(&self) -> io::Result<()> {
        todo!()
    }

    pub fn receive(&self) -> io::Result<()> {
        todo!()
    }

    pub fn close(&self, abort_on_close: bool) -> io::Result<()> {
        let protocol = self.protocol.as_ptr();

        let close_event = uefi::thread::Event::create(
            uefi::raw::EVT_NOTIFY_SIGNAL,
            uefi::raw::TPL_CALLBACK,
            Some(nop_notify),
            None,
        )?;
        let completion_token =
            tcp6::CompletionToken { event: close_event.as_raw_event(), status: Status::ABORTED };
        let mut close_token = tcp6::CloseToken {
            abort_on_close: uefi::raw::Boolean::from(abort_on_close),
            completion_token,
        };
        let r = unsafe { ((*protocol).close)(protocol, &mut close_token) };

        if r.is_error() {
            return match r {
                Status::NOT_STARTED => Err(io::Error::new(
                    io::ErrorKind::Other,
                    "This EFI TCPv6 Protocol instance has not been configured",
                )),
                Status::ACCESS_DENIED => {
                    Err(io::Error::new(io::ErrorKind::PermissionDenied, "EFI_ACCESS_DENIED"))
                }
                Status::INVALID_PARAMETER => {
                    Err(io::Error::new(io::ErrorKind::InvalidInput, "EFI_INVALID_PARAMETER"))
                }
                Status::OUT_OF_RESOURCES => Err(io::Error::new(
                    io::ErrorKind::OutOfMemory,
                    "Could not allocate enough resource to finish the operation",
                )),
                Status::DEVICE_ERROR => {
                    Err(io::Error::new(io::ErrorKind::NetworkDown, "EFI_DEVICE_ERROR"))
                }
                _ => Err(io::Error::new(
                    io::ErrorKind::Uncategorized,
                    format!("Status: {}", r.as_usize()),
                )),
            };
        }

        close_event.wait()?;

        let r = close_token.completion_token.status;
        if r.is_error() {
            match r {
                Status::ABORTED => Err(io::Error::new(
                    io::ErrorKind::ConnectionAborted,
                    "The accept request has been aborted",
                )),
                Status::SECURITY_VIOLATION => Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "The accept operation was failed because of IPSec policy check",
                )),
                _ => Err(io::Error::new(io::ErrorKind::Other, format!("Status: {}", r.as_usize()))),
            }
        } else {
            Ok(())
        }
    }

    pub fn remote_socket(&self) -> io::Result<SocketAddrV6> {
        let config_data = self.get_config_data()?;
        Ok(SocketAddrV6::new(
            Ipv6Addr::from(config_data.access_point.remote_address),
            config_data.access_point.remote_port,
            0,
            0,
        ))
    }

    pub fn station_socket(&self) -> io::Result<SocketAddrV6> {
        let config_data = self.get_config_data()?;
        Ok(SocketAddrV6::new(
            Ipv6Addr::from(config_data.access_point.station_address),
            config_data.access_point.station_port,
            0,
            0,
        ))
    }
}

impl Drop for Tcp6Protocol {
    fn drop(&mut self) {
        let _ = self.service_binding.destroy_child(self.child_handle);
    }
}

#[no_mangle]
pub extern "efiapi" fn nop_notify(_: uefi::raw::Event, _: *mut crate::ffi::c_void) {}
