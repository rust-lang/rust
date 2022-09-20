use super::uefi_service_binding::ServiceBinding;
use super::{ipv4_from_r_efi, ipv4_to_r_efi};
use crate::io::{self, IoSlice, IoSliceMut};
use crate::mem::MaybeUninit;
use crate::net::SocketAddrV4;
use crate::ptr::{addr_of_mut, NonNull};
use crate::sys::uefi::{
    alloc::POOL_ALIGNMENT,
    common::{self, status_to_io_error, VariableBox},
};
use r_efi::efi::Status;
use r_efi::protocols::{ip4, managed_network, simple_network, tcp4};

// FIXME: Discuss what the values these constants should have
const TYPE_OF_SERVICE: u8 = 8;
const TIME_TO_LIVE: u8 = 255;

pub(crate) struct Tcp4Protocol {
    protocol: NonNull<tcp4::Protocol>,
    service_binding: ServiceBinding,
    child_handle: NonNull<crate::ffi::c_void>,
}

impl Tcp4Protocol {
    pub(crate) fn create(service_binding: ServiceBinding) -> io::Result<Tcp4Protocol> {
        let child_handle = service_binding.create_child()?;
        Self::with_child_handle(service_binding, child_handle)
    }

    pub(crate) fn default_config(
        &self,
        use_default_address: bool,
        active_flag: bool,
        station_addr: &crate::net::SocketAddrV4,
        subnet_mask: &crate::net::Ipv4Addr,
        remote_addr: &crate::net::SocketAddrV4,
    ) -> io::Result<()> {
        let mut config_data = tcp4::ConfigData {
            // FIXME: Check in mailing list what traffic_class should be used
            type_of_service: TYPE_OF_SERVICE,
            // FIXME: Check in mailing list what hop_limit should be used
            time_to_live: TIME_TO_LIVE,
            access_point: tcp4::AccessPoint {
                use_default_address: r_efi::efi::Boolean::from(use_default_address),
                station_address: ipv4_to_r_efi(station_addr.ip()),
                station_port: station_addr.port(),
                subnet_mask: ipv4_to_r_efi(subnet_mask),
                remote_address: ipv4_to_r_efi(remote_addr.ip()),
                remote_port: remote_addr.port(),
                active_flag: r_efi::efi::Boolean::from(active_flag),
            },
            // FIXME: Maybe provide a rust default one at some point
            control_option: crate::ptr::null_mut(),
        };
        self.configure(&mut config_data)
    }

    pub(crate) fn configure(&self, config: &mut tcp4::ConfigData) -> io::Result<()> {
        unsafe { Self::config_raw(self.protocol.as_ptr(), config) }
    }

    #[inline]
    pub(crate) fn reset(&self) -> io::Result<()> {
        unsafe { Self::config_raw(self.protocol.as_ptr(), crate::ptr::null_mut()) }
    }

    pub(crate) fn accept(&self) -> io::Result<Tcp4Protocol> {
        let accept_event = common::Event::create(
            r_efi::efi::EVT_NOTIFY_WAIT,
            r_efi::efi::TPL_CALLBACK,
            Some(nop_notify4),
            None,
        )?;
        let completion_token =
            tcp4::CompletionToken { event: accept_event.as_raw_event(), status: Status::ABORTED };

        let mut listen_token =
            tcp4::ListenToken { completion_token, new_child_handle: crate::ptr::null_mut() };

        unsafe { Self::accept_raw(self.protocol.as_ptr(), &mut listen_token) }?;

        accept_event.wait()?;

        let r = listen_token.completion_token.status;
        if r.is_error() {
            Err(status_to_io_error(r))
        } else {
            let child_handle = NonNull::new(listen_token.new_child_handle)
                .ok_or(io::const_io_error!(io::ErrorKind::Other, "null child handle"))?;
            Self::with_child_handle(self.service_binding, child_handle)
        }
    }

    pub(crate) fn connect(&self) -> io::Result<()> {
        let connect_event = common::Event::create(
            r_efi::efi::EVT_NOTIFY_WAIT,
            r_efi::efi::TPL_CALLBACK,
            Some(nop_notify4),
            None,
        )?;
        let completion_token =
            tcp4::CompletionToken { event: connect_event.as_raw_event(), status: Status::ABORTED };
        let mut connection_token = tcp4::ConnectionToken { completion_token };
        unsafe { Self::connect_raw(self.protocol.as_ptr(), &mut connection_token) }?;
        connect_event.wait()?;
        let r = connection_token.completion_token.status;
        if r.is_error() { Err(status_to_io_error(r)) } else { Ok(()) }
    }

    pub(crate) fn transmit(&self, buf: &[u8]) -> io::Result<usize> {
        let buf_size = buf.len() as u32;
        let transmit_event = common::Event::create(
            r_efi::efi::EVT_NOTIFY_WAIT,
            r_efi::efi::TPL_CALLBACK,
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

        let mut transmit_data = TransmitData {
            push: r_efi::efi::Boolean::TRUE,
            urgent: r_efi::efi::Boolean::FALSE,
            data_length: buf_size,
            fragment_count: 1,
            fragment_table: [fragment_table; 1],
        };

        let packet = tcp4::IoTokenPacket {
            tx_data: &mut transmit_data as *mut TransmitData<1> as *mut tcp4::TransmitData,
        };
        let mut transmit_token = tcp4::IoToken { completion_token, packet };
        unsafe { Self::transmit_raw(self.protocol.as_ptr(), &mut transmit_token) }?;

        transmit_event.wait()?;

        let r = transmit_token.completion_token.status;
        if r.is_error() {
            Err(status_to_io_error(r))
        } else {
            Ok(unsafe { (*transmit_token.packet.tx_data).data_length } as usize)
        }
    }

    pub(crate) fn transmit_vectored(&self, buf: &[IoSlice<'_>]) -> io::Result<usize> {
        let buf_size = crate::mem::size_of_val(buf);
        let transmit_event = common::Event::create(
            r_efi::efi::EVT_NOTIFY_WAIT,
            r_efi::efi::TPL_CALLBACK,
            Some(nop_notify4),
            None,
        )?;
        let completion_token =
            tcp4::CompletionToken { event: transmit_event.as_raw_event(), status: Status::ABORTED };
        let fragment_tables: Vec<tcp4::FragmentData> = buf
            .iter()
            .map(|b| tcp4::FragmentData {
                fragment_length: b.len() as u32,
                fragment_buffer: (*b).as_ptr() as *mut crate::ffi::c_void,
            })
            .collect();

        let layout = unsafe {
            crate::alloc::Layout::from_size_align_unchecked(
                crate::mem::size_of::<tcp4::TransmitData>()
                    + crate::mem::size_of_val(&fragment_tables),
                POOL_ALIGNMENT,
            )
        };
        let mut transmit_data = VariableBox::<TransmitData<0>>::new_uninit(layout);
        let fragment_tables_len = fragment_tables.len();

        // Initialize TransmitData
        unsafe {
            addr_of_mut!((*transmit_data.as_uninit_mut_ptr()).push)
                .write(r_efi::efi::Boolean::TRUE);
            addr_of_mut!((*transmit_data.as_uninit_mut_ptr()).urgent)
                .write(r_efi::efi::Boolean::FALSE);
            addr_of_mut!((*transmit_data.as_uninit_mut_ptr()).data_length).write(buf_size as u32);
            addr_of_mut!((*transmit_data.as_uninit_mut_ptr()).fragment_count)
                .write(fragment_tables_len as u32);
            addr_of_mut!((*transmit_data.as_uninit_mut_ptr()).fragment_table)
                .cast::<tcp4::FragmentData>()
                .copy_from_nonoverlapping(fragment_tables.as_ptr(), fragment_tables_len);
        };
        let mut transmit_data = unsafe { transmit_data.assume_init() };

        let packet = tcp4::IoTokenPacket { tx_data: transmit_data.as_mut_ptr().cast() };
        let mut transmit_token = tcp4::IoToken { completion_token, packet };
        unsafe { Self::transmit_raw(self.protocol.as_ptr(), &mut transmit_token) }?;

        transmit_event.wait()?;

        let r = transmit_token.completion_token.status;
        if r.is_error() {
            Err(status_to_io_error(r))
        } else {
            Ok(unsafe { (*transmit_token.packet.tx_data).data_length } as usize)
        }
    }

    pub(crate) fn receive(&self, buf: &mut [u8]) -> io::Result<usize> {
        let buf_size = buf.len() as u32;
        let receive_event = common::Event::create(
            r_efi::efi::EVT_NOTIFY_WAIT,
            r_efi::efi::TPL_CALLBACK,
            Some(nop_notify4),
            None,
        )?;
        let fragment_table = tcp4::FragmentData {
            fragment_length: buf_size,
            fragment_buffer: buf.as_mut_ptr().cast(),
        };

        let mut receive_data = ReceiveData {
            urgent_flag: r_efi::efi::Boolean::FALSE,
            data_length: buf_size,
            fragment_count: 1,
            fragment_table: [fragment_table; 1],
        };

        let packet = tcp4::IoTokenPacket {
            rx_data: &mut receive_data as *mut ReceiveData<1> as *mut tcp4::ReceiveData,
        };
        let completion_token =
            tcp4::CompletionToken { event: receive_event.as_raw_event(), status: Status::ABORTED };
        let mut receive_token = tcp4::IoToken { completion_token, packet };
        unsafe { Self::receive_raw(self.protocol.as_ptr(), &mut receive_token) }?;

        receive_event.wait()?;

        let r = receive_token.completion_token.status;
        if r.is_error() {
            Err(status_to_io_error(r))
        } else {
            Ok(unsafe { (*receive_token.packet.rx_data).data_length } as usize)
        }
    }

    pub(crate) fn receive_vectored(&self, buf: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        let receive_event = common::Event::create(
            r_efi::efi::EVT_NOTIFY_WAIT,
            r_efi::efi::TPL_CALLBACK,
            Some(nop_notify4),
            None,
        )?;

        let buf_size = crate::mem::size_of_val(&buf) as u32;
        let fragment_tables: Vec<tcp4::FragmentData> = buf
            .iter_mut()
            .map(|b| tcp4::FragmentData {
                fragment_length: b.len() as u32,
                fragment_buffer: b.as_mut_ptr().cast(),
            })
            .collect();
        let fragment_tables_len = fragment_tables.len();

        let layout = unsafe {
            crate::alloc::Layout::from_size_align_unchecked(
                crate::mem::size_of::<tcp4::ReceiveData>()
                    + crate::mem::size_of_val(&fragment_tables),
                POOL_ALIGNMENT,
            )
        };
        let mut receive_data = VariableBox::<ReceiveData<0>>::new_uninit(layout);
        unsafe {
            addr_of_mut!((*receive_data.as_uninit_mut_ptr()).urgent_flag)
                .write(r_efi::efi::Boolean::FALSE);
            addr_of_mut!((*receive_data.as_uninit_mut_ptr()).data_length).write(buf_size);
            addr_of_mut!((*receive_data.as_uninit_mut_ptr()).fragment_count)
                .write(fragment_tables_len as u32);
            addr_of_mut!((*receive_data.as_uninit_mut_ptr()).fragment_table)
                .cast::<tcp4::FragmentData>()
                .copy_from_nonoverlapping(fragment_tables.as_ptr(), fragment_tables_len);
        }
        let mut receive_data = unsafe { receive_data.assume_init() };

        let packet = tcp4::IoTokenPacket { rx_data: receive_data.as_mut_ptr().cast() };
        let completion_token =
            tcp4::CompletionToken { event: receive_event.as_raw_event(), status: Status::ABORTED };
        let mut receive_token = tcp4::IoToken { completion_token, packet };
        unsafe { Self::receive_raw(self.protocol.as_ptr(), &mut receive_token) }?;

        receive_event.wait()?;

        let r = receive_token.completion_token.status;
        if r.is_error() {
            Err(status_to_io_error(r))
        } else {
            Ok(unsafe { (*receive_token.packet.rx_data).data_length } as usize)
        }
    }

    pub(crate) fn close(&self, abort_on_close: bool) -> io::Result<()> {
        let protocol = self.protocol.as_ptr();

        let close_event = common::Event::create(
            r_efi::efi::EVT_NOTIFY_WAIT,
            r_efi::efi::TPL_CALLBACK,
            Some(nop_notify4),
            None,
        )?;
        let completion_token =
            tcp4::CompletionToken { event: close_event.as_raw_event(), status: Status::ABORTED };
        let mut close_token = tcp4::CloseToken {
            abort_on_close: r_efi::efi::Boolean::from(abort_on_close),
            completion_token,
        };
        let r = unsafe { ((*protocol).close)(protocol, &mut close_token) };

        if r.is_error() {
            return Err(status_to_io_error(r));
        }

        close_event.wait()?;

        let r = close_token.completion_token.status;
        if r.is_error() { Err(status_to_io_error(r)) } else { Ok(()) }
    }

    pub(crate) fn remote_socket(&self) -> io::Result<SocketAddrV4> {
        let config_data = self.get_config_data()?;
        Ok(SocketAddrV4::new(
            ipv4_from_r_efi(config_data.access_point.remote_address),
            config_data.access_point.remote_port,
        ))
    }

    pub(crate) fn station_socket(&self) -> io::Result<SocketAddrV4> {
        let config_data = self.get_config_data()?;
        Ok(SocketAddrV4::new(
            ipv4_from_r_efi(config_data.access_point.station_address),
            config_data.access_point.station_port,
        ))
    }

    #[inline]
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
        let tcp4_protocol = common::open_protocol(child_handle, tcp4::PROTOCOL_GUID)?;
        Ok(Self::new(tcp4_protocol, service_binding, child_handle))
    }

    pub(crate) fn get_config_data(&self) -> io::Result<tcp4::ConfigData> {
        // Using MaybeUninit::uninit() generates a Page Fault Here
        let mut config_data: MaybeUninit<tcp4::ConfigData> = MaybeUninit::zeroed();
        unsafe {
            Self::get_mode_data_raw(
                self.protocol.as_ptr(),
                crate::ptr::null_mut(),
                config_data.as_mut_ptr(),
                crate::ptr::null_mut(),
                crate::ptr::null_mut(),
                crate::ptr::null_mut(),
            )
        }?;
        Ok(unsafe { config_data.assume_init() })
    }

    unsafe fn receive_raw(
        protocol: *mut tcp4::Protocol,
        token: *mut tcp4::IoToken,
    ) -> io::Result<()> {
        let r = unsafe { ((*protocol).receive)(protocol, token) };
        if r.is_error() { Err(status_to_io_error(r)) } else { Ok(()) }
    }

    unsafe fn transmit_raw(
        protocol: *mut tcp4::Protocol,
        token: *mut tcp4::IoToken,
    ) -> io::Result<()> {
        let r = unsafe { ((*protocol).transmit)(protocol, token) };
        if r.is_error() { Err(status_to_io_error(r)) } else { Ok(()) }
    }

    unsafe fn config_raw(
        protocol: *mut tcp4::Protocol,
        config_data: *mut tcp4::ConfigData,
    ) -> io::Result<()> {
        let r = unsafe { ((*protocol).configure)(protocol, config_data) };
        if r.is_error() { Err(status_to_io_error(r)) } else { Ok(()) }
    }

    unsafe fn accept_raw(
        protocol: *mut tcp4::Protocol,
        token: *mut tcp4::ListenToken,
    ) -> io::Result<()> {
        let r = unsafe { ((*protocol).accept)(protocol, token) };
        if r.is_error() { Err(status_to_io_error(r)) } else { Ok(()) }
    }

    unsafe fn get_mode_data_raw(
        protocol: *mut tcp4::Protocol,
        tcp4_state: *mut tcp4::ConnectionState,
        tcp4_config_data: *mut tcp4::ConfigData,
        ip4_mode_data: *mut ip4::ModeData,
        mnp_config_data: *mut managed_network::ConfigData,
        snp_mode_data: *mut simple_network::Mode,
    ) -> io::Result<()> {
        let r = unsafe {
            ((*protocol).get_mode_data)(
                protocol,
                tcp4_state,
                tcp4_config_data,
                ip4_mode_data,
                mnp_config_data,
                snp_mode_data,
            )
        };
        if r.is_error() { Err(status_to_io_error(r)) } else { Ok(()) }
    }

    unsafe fn connect_raw(
        protocol: *mut tcp4::Protocol,
        token: *mut tcp4::ConnectionToken,
    ) -> io::Result<()> {
        let r = unsafe { ((*protocol).connect)(protocol, token) };
        if r.is_error() { Err(status_to_io_error(r)) } else { Ok(()) }
    }
}

impl Drop for Tcp4Protocol {
    fn drop(&mut self) {
        let _ = self.close(true);
        let _ = self.service_binding.destroy_child(self.child_handle);
    }
}

#[no_mangle]
extern "efiapi" fn nop_notify4(_: r_efi::efi::Event, _: *mut crate::ffi::c_void) {}

// Safety: No one besides us has the raw pointer (since the handle was created using the Service binding Protocol).
// Also there are no threads to transfer the pointer to.
unsafe impl Send for Tcp4Protocol {}

// Safety: There are no threads in UEFI
unsafe impl Sync for Tcp4Protocol {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TransmitData<const N: usize> {
    pub push: r_efi::efi::Boolean,
    pub urgent: r_efi::efi::Boolean,
    pub data_length: u32,
    pub fragment_count: u32,
    pub fragment_table: [tcp4::FragmentData; N],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReceiveData<const N: usize> {
    pub urgent_flag: r_efi::efi::Boolean,
    pub data_length: u32,
    pub fragment_count: u32,
    pub fragment_table: [tcp4::FragmentData; N],
}
