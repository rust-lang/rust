use super::uefi_service_binding::{self, ServiceBinding};
use crate::io;
use crate::mem::MaybeUninit;
use crate::net::{Ipv4Addr, SocketAddrV4};
use crate::os::uefi;
use crate::os::uefi::raw::protocols::{
    ip4, managed_network, service_binding, simple_network, tcp4,
};
use crate::os::uefi::raw::Status;
use crate::ptr::NonNull;
use crate::sys_common::AsInner;

// FIXME: Discuss what the values these constants should have
const TYPE_OF_SERVICE: u8 = 8;
const TIME_TO_LIVE: u8 = 255;

pub struct Tcp4Protocol {
    protocol: NonNull<tcp4::Protocol>,
    service_binding: ServiceBinding,
    child_handle: NonNull<crate::ffi::c_void>,
}

impl Tcp4Protocol {
    pub fn create(service_binding: ServiceBinding) -> io::Result<Tcp4Protocol> {
        let child_handle = service_binding.create_child()?;
        Self::with_child_handle(service_binding, child_handle)
    }

    pub fn config(
        &self,
        use_default_address: bool,
        active_flag: bool,
        station_addr: &crate::net::SocketAddrV4,
        subnet_mask: &crate::net::Ipv4Addr,
        remote_addr: &crate::net::SocketAddrV4,
    ) -> io::Result<()> {
        let protocol = self.protocol.as_ptr();

        let mut config_data = tcp4::ConfigData {
            // FIXME: Check in mailing list what traffic_class should be used
            type_of_service: TYPE_OF_SERVICE,
            // FIXME: Check in mailing list what hop_limit should be used
            time_to_live: TIME_TO_LIVE,
            access_point: tcp4::AccessPoint {
                use_default_address: uefi::raw::Boolean::from(use_default_address),
                station_address: uefi::raw::Ipv4Address::from(station_addr.ip()),
                station_port: station_addr.port(),
                subnet_mask: uefi::raw::Ipv4Address::from(subnet_mask),
                remote_address: uefi::raw::Ipv4Address::from(remote_addr.ip()),
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
                _ => io::Error::new(
                    io::ErrorKind::Uncategorized,
                    format!("Unknown Error: {}", r.as_usize()),
                ),
            };
            Err(e)
        } else {
            Ok(())
        }
    }

    pub fn accept(&self) -> io::Result<Tcp4Protocol> {
        let protocol = self.protocol.as_ptr();

        let accept_event = uefi::thread::Event::create(
            uefi::raw::EVT_NOTIFY_WAIT,
            uefi::raw::TPL_CALLBACK,
            Some(nop_notify4),
            None,
        )?;
        let completion_token =
            tcp4::CompletionToken { event: accept_event.as_raw_event(), status: Status::ABORTED };

        let mut listen_token = tcp4::ListenToken {
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
        accept_event.wait()?;

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

    pub fn transmit(&self, buf: &[u8]) -> io::Result<usize> {
        let buf_size = buf.len() as u32;
        let transmit_event = uefi::thread::Event::create(
            uefi::raw::EVT_NOTIFY_WAIT,
            uefi::raw::TPL_CALLBACK,
            Some(nop_notify4),
            None,
        )?;
        let completion_token =
            tcp4::CompletionToken { event: transmit_event.as_raw_event(), status: Status::ABORTED };
        let fragment_table = tcp4::FragmentData {
            fragment_length: buf_size,
            // FIXME: Probably dangerous
            fragment_buffer: buf.as_ptr() as *mut crate::ffi::c_void,
        };
        let mut transmit_data = tcp4::TransmitData {
            push: uefi::raw::Boolean::from(true),
            urgent: uefi::raw::Boolean::from(false),
            data_length: buf_size,
            fragment_count: 1,
            fragment_table: [],
        };

        unsafe { transmit_data.fragment_table.as_mut_ptr().swap([fragment_table].as_mut_ptr()) };

        let packet = tcp4::IoTokenPacket { tx_data: &mut transmit_data };
        let mut transmit_token = tcp4::IoToken { completion_token, packet };
        Self::transmit_raw(self.protocol.as_ptr(), &mut transmit_token)?;

        transmit_event.wait()?;

        let r = transmit_token.completion_token.status;
        if r.is_error() {
            match r {
                Status::CONNECTION_FIN => {
                    Err(io::Error::new(io::ErrorKind::Other, "EFI_CONNECTION_FIN"))
                }
                Status::CONNECTION_RESET => {
                    Err(io::Error::new(io::ErrorKind::ConnectionReset, "EFI_CONNECTION_RESET"))
                }
                Status::ABORTED => {
                    Err(io::Error::new(io::ErrorKind::ConnectionAborted, "EFI_ABORTED"))
                }
                Status::TIMEOUT => Err(io::Error::new(io::ErrorKind::TimedOut, "EFI_TIMEOUT")),
                // Status::NETWORK_UNREACHABLE => Err(io::Error::new(io::ErrorKind::TimedOut, "EFI_TIMEOUT")),
                // Status::HOST_UNREACHABLE => Err(io::Error::new(io::ErrorKind::TimedOut, "EFI_TIMEOUT")),
                // Status::PROTOCOL_UNREACHABLE => Err(io::Error::new(io::ErrorKind::TimedOut, "EFI_TIMEOUT")),
                // Status::PORT_UNREACHABLE => Err(io::Error::new(io::ErrorKind::TimedOut, "EFI_TIMEOUT")),
                Status::ICMP_ERROR => Err(io::Error::new(io::ErrorKind::Other, "EFI_ICMP_ERROR")),
                Status::DEVICE_ERROR => {
                    Err(io::Error::new(io::ErrorKind::Other, "EFI_DEVICE_ERROR"))
                }
                Status::NO_MEDIA => Err(io::Error::new(io::ErrorKind::Other, "EFI_NO_MEDIA")),
                _ => Err(io::Error::new(io::ErrorKind::Other, format!("Status: {}", r.as_usize()))),
            }
        } else {
            Ok(unsafe { (*transmit_token.packet.tx_data).data_length } as usize)
        }
    }

    pub fn receive(&self, buf: &mut [u8]) -> io::Result<usize> {
        let buf_size = buf.len() as u32;
        let receive_event = uefi::thread::Event::create(
            uefi::raw::EVT_NOTIFY_WAIT,
            uefi::raw::TPL_CALLBACK,
            Some(nop_notify4),
            None,
        )?;
        let fragment_table = tcp4::FragmentData {
            fragment_length: buf_size,
            fragment_buffer: buf.as_mut_ptr().cast(),
        };
        let mut receive_data = tcp4::ReceiveData {
            urgent_flag: uefi::raw::Boolean::from(false),
            data_length: buf_size,
            fragment_count: 1,
            fragment_table: [],
        };

        unsafe { receive_data.fragment_table.as_mut_ptr().swap([fragment_table].as_mut_ptr()) };

        let packet = tcp4::IoTokenPacket { rx_data: &mut receive_data };
        let completion_token =
            tcp4::CompletionToken { event: receive_event.as_raw_event(), status: Status::ABORTED };
        let mut receive_token = tcp4::IoToken { completion_token, packet };
        Self::receive_raw(self.protocol.as_ptr(), &mut receive_token)?;

        println!("Wait for receive");
        receive_event.wait()?;
        println!("Receive Done");

        let r = receive_token.completion_token.status;
        if r.is_error() {
            match r {
                Status::CONNECTION_FIN => {
                    Err(io::Error::new(io::ErrorKind::Other, "EFI_CONNECTION_FIN"))
                }
                Status::CONNECTION_RESET => {
                    Err(io::Error::new(io::ErrorKind::ConnectionReset, "EFI_CONNECTION_RESET"))
                }
                Status::ABORTED => {
                    Err(io::Error::new(io::ErrorKind::ConnectionAborted, "EFI_ABORTED"))
                }
                Status::TIMEOUT => Err(io::Error::new(io::ErrorKind::TimedOut, "EFI_TIMEOUT")),
                // Status::NETWORK_UNREACHABLE => Err(io::Error::new(io::ErrorKind::TimedOut, "EFI_TIMEOUT")),
                // Status::HOST_UNREACHABLE => Err(io::Error::new(io::ErrorKind::TimedOut, "EFI_TIMEOUT")),
                // Status::PROTOCOL_UNREACHABLE => Err(io::Error::new(io::ErrorKind::TimedOut, "EFI_TIMEOUT")),
                // Status::PORT_UNREACHABLE => Err(io::Error::new(io::ErrorKind::TimedOut, "EFI_TIMEOUT")),
                Status::ICMP_ERROR => Err(io::Error::new(io::ErrorKind::Other, "EFI_ICMP_ERROR")),
                Status::DEVICE_ERROR => {
                    Err(io::Error::new(io::ErrorKind::Other, "EFI_DEVICE_ERROR"))
                }
                Status::NO_MEDIA => Err(io::Error::new(io::ErrorKind::Other, "EFI_NO_MEDIA")),
                _ => Err(io::Error::new(io::ErrorKind::Other, format!("Status: {}", r.as_usize()))),
            }
        } else {
            Ok(unsafe { (*receive_token.packet.rx_data).data_length } as usize)
        }
    }

    pub fn close(&self, abort_on_close: bool) -> io::Result<()> {
        let protocol = self.protocol.as_ptr();

        let close_event = uefi::thread::Event::create(
            uefi::raw::EVT_NOTIFY_WAIT,
            uefi::raw::TPL_CALLBACK,
            Some(nop_notify4),
            None,
        )?;
        let completion_token =
            tcp4::CompletionToken { event: close_event.as_raw_event(), status: Status::ABORTED };
        let mut close_token = tcp4::CloseToken {
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

    pub fn remote_socket(&self) -> io::Result<SocketAddrV4> {
        let config_data = self.get_config_data()?;
        Ok(SocketAddrV4::new(
            Ipv4Addr::from(config_data.access_point.remote_address),
            config_data.access_point.remote_port,
        ))
    }

    pub fn station_socket(&self) -> io::Result<SocketAddrV4> {
        let config_data = self.get_config_data()?;
        Ok(SocketAddrV4::new(
            Ipv4Addr::from(config_data.access_point.station_address),
            config_data.access_point.station_port,
        ))
    }

    fn new(
        protocol: NonNull<tcp4::Protocol>,
        service_binding: ServiceBinding,
        child_handle: NonNull<crate::ffi::c_void>,
    ) -> Self {
        Self { protocol, service_binding, child_handle }
    }

    fn with_child_handle(
        service_binding: ServiceBinding,
        child_handle: NonNull<crate::ffi::c_void>,
    ) -> io::Result<Self> {
        let tcp4_protocol = uefi::env::open_protocol(child_handle, tcp4::PROTOCOL_GUID)?;
        Ok(Self::new(tcp4_protocol, service_binding, child_handle))
    }

    // FIXME: This function causes the program to freeze.
    fn get_config_data(&self) -> io::Result<tcp4::ConfigData> {
        let protocol = self.protocol.as_ptr();

        let mut state: MaybeUninit<tcp4::ConnectionState> = MaybeUninit::uninit();
        let mut config_data: MaybeUninit<tcp4::ConfigData> = MaybeUninit::uninit();
        let mut ip4_mode_data: MaybeUninit<ip4::ModeData> = MaybeUninit::uninit();
        let mut mnp_mode_data: MaybeUninit<managed_network::ConfigData> = MaybeUninit::uninit();
        let mut snp_mode_data: MaybeUninit<simple_network::Mode> = MaybeUninit::uninit();

        let r = unsafe {
            ((*protocol).get_mode_data)(
                protocol,
                state.as_mut_ptr(),
                config_data.as_mut_ptr(),
                ip4_mode_data.as_mut_ptr(),
                mnp_mode_data.as_mut_ptr(),
                snp_mode_data.as_mut_ptr(),
            )
        };

        if r.is_error() {
            match r {
                Status::NOT_STARTED => Err(io::Error::new(io::ErrorKind::Other, "EFI_NOT_STARTED")),
                Status::INVALID_PARAMETER => {
                    Err(io::Error::new(io::ErrorKind::InvalidInput, "EFI_INVALID_PARAMETER"))
                }
                _ => Err(io::Error::new(
                    io::ErrorKind::Uncategorized,
                    format!("Status: {}", r.as_usize()),
                )),
            }
        } else {
            unsafe {
                state.assume_init_drop();
                ip4_mode_data.assume_init_drop();
                mnp_mode_data.assume_init_drop();
                snp_mode_data.assume_init_drop();
            }
            Ok(unsafe { config_data.assume_init() })
        }
    }

    fn receive_raw(protocol: *mut tcp4::Protocol, token: *mut tcp4::IoToken) -> io::Result<()> {
        let r = unsafe { ((*protocol).receive)(protocol, token) };

        if r.is_error() {
            match r {
                Status::NOT_STARTED => Err(io::Error::new(io::ErrorKind::Other, "EFI_NOT_STARTED")),
                Status::NO_MAPPING => Err(io::Error::new(io::ErrorKind::Other, "EFI_NO_MAPPING")),
                Status::INVALID_PARAMETER => {
                    Err(io::Error::new(io::ErrorKind::InvalidInput, "EFI_INVALID_PARAMETER"))
                }
                Status::OUT_OF_RESOURCES => {
                    Err(io::Error::new(io::ErrorKind::Other, "EFI_OUT_OF_RESOURCES"))
                }
                Status::DEVICE_ERROR => {
                    Err(io::Error::new(io::ErrorKind::Other, "EFI_DEVICE_ERROR"))
                }
                Status::ACCESS_DENIED => {
                    Err(io::Error::new(io::ErrorKind::PermissionDenied, "EFI_ACCESS_DENIED"))
                }
                Status::CONNECTION_FIN => {
                    Err(io::Error::new(io::ErrorKind::Other, "EFI_CONNECTION_FIN"))
                }
                Status::NOT_READY => Err(io::Error::new(io::ErrorKind::Other, "EFI_NOT_READY")),
                Status::NO_MEDIA => Err(io::Error::new(io::ErrorKind::Other, "EFI_NO_MEDIA")),
                _ => Err(io::Error::new(
                    io::ErrorKind::Uncategorized,
                    format!("Status: {}", r.as_usize()),
                )),
            }
        } else {
            Ok(())
        }
    }

    fn transmit_raw(protocol: *mut tcp4::Protocol, token: *mut tcp4::IoToken) -> io::Result<()> {
        let r = unsafe { ((*protocol).transmit)(protocol, token) };

        if r.is_error() {
            match r {
                Status::NOT_STARTED => Err(io::Error::new(io::ErrorKind::Other, "EFI_NOT_STARTED")),
                Status::NO_MAPPING => Err(io::Error::new(io::ErrorKind::Other, "EFI_NO_MAPPING")),
                Status::INVALID_PARAMETER => {
                    Err(io::Error::new(io::ErrorKind::InvalidInput, "EFI_INVALID_PARAMETER"))
                }
                Status::OUT_OF_RESOURCES => {
                    Err(io::Error::new(io::ErrorKind::Other, "EFI_OUT_OF_RESOURCES"))
                }
                Status::DEVICE_ERROR => {
                    Err(io::Error::new(io::ErrorKind::Other, "EFI_DEVICE_ERROR"))
                }
                Status::ACCESS_DENIED => {
                    Err(io::Error::new(io::ErrorKind::PermissionDenied, "EFI_ACCESS_DENIED"))
                }
                Status::NO_MEDIA => Err(io::Error::new(io::ErrorKind::Other, "EFI_NO_MEDIA")),
                Status::NOT_READY => Err(io::Error::new(io::ErrorKind::Other, "EFI_NOT_READY")),
                // Status::NETWORK_UNREACHABLE => {
                //     Err(io::Error::new(io::ErrorKind::Other, "EFI_NETWORK_UNREACHABLE"))
                // }
                _ => Err(io::Error::new(
                    io::ErrorKind::Uncategorized,
                    format!("Status: {}", r.as_usize()),
                )),
            }
        } else {
            Ok(())
        }
    }
}

impl Drop for Tcp4Protocol {
    fn drop(&mut self) {
        let _ = self.close(true);
        let _ = self.service_binding.destroy_child(self.child_handle);
    }
}

#[no_mangle]
pub extern "efiapi" fn nop_notify4(_: uefi::raw::Event, _: *mut crate::ffi::c_void) {}
