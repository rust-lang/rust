use r_efi::efi::{self, Status};
use r_efi::protocols::tcp4;

use crate::io::{self, IoSlice, IoSliceMut};
use crate::net::SocketAddrV4;
use crate::ptr::{self, NonNull};
use crate::sync::atomic::{AtomicBool, Ordering};
use crate::sys::pal::helpers;
use crate::time::{Duration, Instant};

const TYPE_OF_SERVICE: u8 = 8;
const TIME_TO_LIVE: u8 = 255;

pub(crate) struct Tcp4 {
    handle: NonNull<crate::ffi::c_void>,
    protocol: NonNull<tcp4::Protocol>,
    flag: AtomicBool,
    service_binding: helpers::ServiceProtocol,
}

const DEFAULT_ADDR: efi::Ipv4Address = efi::Ipv4Address { addr: [0u8; 4] };

impl Tcp4 {
    pub(crate) fn new() -> io::Result<Self> {
        let (service_binding, handle) =
            helpers::ServiceProtocol::open(tcp4::SERVICE_BINDING_PROTOCOL_GUID)?;
        let protocol = helpers::open_protocol(handle, tcp4::PROTOCOL_GUID)?;

        Ok(Self { service_binding, handle, protocol, flag: AtomicBool::new(false) })
    }

    pub(crate) fn configure(
        &self,
        active: bool,
        remote_address: Option<&SocketAddrV4>,
        station_address: Option<&SocketAddrV4>,
    ) -> io::Result<()> {
        let protocol = self.protocol.as_ptr();

        let (remote_address, remote_port) = if let Some(x) = remote_address {
            (helpers::ipv4_to_r_efi(*x.ip()), x.port())
        } else {
            (DEFAULT_ADDR, 0)
        };

        let use_default_address: r_efi::efi::Boolean =
            station_address.is_none_or(|addr| addr.ip().is_unspecified()).into();
        let (station_address, station_port) = if let Some(x) = station_address {
            (helpers::ipv4_to_r_efi(*x.ip()), x.port())
        } else {
            (DEFAULT_ADDR, 0)
        };
        let subnet_mask = helpers::ipv4_to_r_efi(crate::net::Ipv4Addr::new(255, 255, 255, 0));

        let mut config_data = tcp4::ConfigData {
            type_of_service: TYPE_OF_SERVICE,
            time_to_live: TIME_TO_LIVE,
            access_point: tcp4::AccessPoint {
                use_default_address,
                remote_address,
                remote_port,
                active_flag: active.into(),
                station_address,
                station_port,
                subnet_mask,
            },
            control_option: ptr::null_mut(),
        };

        let r = unsafe { ((*protocol).configure)(protocol, &mut config_data) };
        if r.is_error() { Err(crate::io::Error::from_raw_os_error(r.as_usize())) } else { Ok(()) }
    }

    pub(crate) fn get_mode_data(&self) -> io::Result<tcp4::ConfigData> {
        let mut config_data = tcp4::ConfigData::default();
        let protocol = self.protocol.as_ptr();

        let r = unsafe {
            ((*protocol).get_mode_data)(
                protocol,
                ptr::null_mut(),
                &mut config_data,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        if r.is_error() { Err(io::Error::from_raw_os_error(r.as_usize())) } else { Ok(config_data) }
    }

    pub(crate) fn accept(&self) -> io::Result<Self> {
        let evt = unsafe { self.create_evt() }?;
        let completion_token =
            tcp4::CompletionToken { event: evt.as_ptr(), status: Status::SUCCESS };
        let mut listen_token =
            tcp4::ListenToken { completion_token, new_child_handle: ptr::null_mut() };

        let protocol = self.protocol.as_ptr();
        let r = unsafe { ((*protocol).accept)(protocol, &mut listen_token) };
        if r.is_error() {
            return Err(io::Error::from_raw_os_error(r.as_usize()));
        }

        unsafe { self.wait_or_cancel(None, &mut listen_token.completion_token) }?;

        if completion_token.status.is_error() {
            Err(io::Error::from_raw_os_error(completion_token.status.as_usize()))
        } else {
            // EDK2 internals seem to assume a single ServiceBinding Protocol for TCP4 and TCP6, and
            // thus does not use any service binding protocol data in destroying child sockets. It
            // does seem to suggest that we need to cleanup even the protocols created by accept. To
            // be on the safe side with other implementations, we will be using the same service
            // binding protocol as the parent TCP4 handle.
            //
            // https://github.com/tianocore/edk2/blob/f80580f56b267c96f16f985dbf707b2f96947da4/NetworkPkg/TcpDxe/TcpDriver.c#L938

            let handle = NonNull::new(listen_token.new_child_handle).unwrap();
            let protocol = helpers::open_protocol(handle, tcp4::PROTOCOL_GUID)?;

            Ok(Self {
                handle,
                service_binding: self.service_binding,
                protocol,
                flag: AtomicBool::new(false),
            })
        }
    }

    pub(crate) fn connect(&self, timeout: Option<Duration>) -> io::Result<()> {
        let evt = unsafe { self.create_evt() }?;
        let completion_token =
            tcp4::CompletionToken { event: evt.as_ptr(), status: Status::SUCCESS };

        let protocol = self.protocol.as_ptr();
        let mut conn_token = tcp4::ConnectionToken { completion_token };

        let r = unsafe { ((*protocol).connect)(protocol, &mut conn_token) };
        if r.is_error() {
            return Err(io::Error::from_raw_os_error(r.as_usize()));
        }

        unsafe { self.wait_or_cancel(timeout, &mut conn_token.completion_token) }?;

        if completion_token.status.is_error() {
            Err(io::Error::from_raw_os_error(completion_token.status.as_usize()))
        } else {
            Ok(())
        }
    }

    pub(crate) fn write(&self, buf: &[u8], timeout: Option<Duration>) -> io::Result<usize> {
        let data_len = u32::try_from(buf.len()).unwrap_or(u32::MAX);
        let fragment = tcp4::FragmentData {
            fragment_length: data_len,
            fragment_buffer: buf.as_ptr().cast::<crate::ffi::c_void>().cast_mut(),
        };
        let mut tx_data = tcp4::TransmitData {
            push: r_efi::efi::Boolean::FALSE,
            urgent: r_efi::efi::Boolean::FALSE,
            data_length: data_len,
            fragment_count: 1,
            fragment_table: [fragment],
        };

        self.write_inner((&raw mut tx_data).cast(), timeout).map(|_| data_len as usize)
    }

    pub(crate) fn write_vectored(
        &self,
        buf: &[IoSlice<'_>],
        timeout: Option<Duration>,
    ) -> io::Result<usize> {
        let mut data_length = 0u32;
        let mut fragment_count = 0u32;

        // Calculate how many IoSlice in buf can be transmitted.
        for i in buf {
            // IoSlice length is always <= u32::MAX in UEFI.
            match data_length
                .checked_add(u32::try_from(i.as_slice().len()).expect("value is stored as a u32"))
            {
                Some(x) => data_length = x,
                None => break,
            }
            fragment_count += 1;
        }

        let tx_data_size = size_of::<tcp4::TransmitData<0>>()
            + size_of::<tcp4::FragmentData>() * (fragment_count as usize);
        let mut tx_data = helpers::UefiBox::<tcp4::TransmitData>::new(tx_data_size)?;
        tx_data.write(tcp4::TransmitData {
            push: r_efi::efi::Boolean::FALSE,
            urgent: r_efi::efi::Boolean::FALSE,
            data_length,
            fragment_count,
            fragment_table: [],
        });
        unsafe {
            // SAFETY: IoSlice and FragmentData are guaranteed to have same layout.
            crate::ptr::copy_nonoverlapping(
                buf.as_ptr().cast(),
                (*tx_data.as_mut_ptr()).fragment_table.as_mut_ptr(),
                fragment_count as usize,
            );
        };

        self.write_inner(tx_data.as_mut_ptr(), timeout).map(|_| data_length as usize)
    }

    fn write_inner(
        &self,
        tx_data: *mut tcp4::TransmitData,
        timeout: Option<Duration>,
    ) -> io::Result<()> {
        let evt = unsafe { self.create_evt() }?;
        let completion_token =
            tcp4::CompletionToken { event: evt.as_ptr(), status: Status::SUCCESS };

        let protocol = self.protocol.as_ptr();
        let mut token = tcp4::IoToken { completion_token, packet: tcp4::IoTokenPacket { tx_data } };

        let r = unsafe { ((*protocol).transmit)(protocol, &mut token) };
        if r.is_error() {
            return Err(io::Error::from_raw_os_error(r.as_usize()));
        }

        unsafe { self.wait_or_cancel(timeout, &mut token.completion_token) }?;

        if completion_token.status.is_error() {
            Err(io::Error::from_raw_os_error(completion_token.status.as_usize()))
        } else {
            Ok(())
        }
    }

    pub(crate) fn read(&self, buf: &mut [u8], timeout: Option<Duration>) -> io::Result<usize> {
        let data_len = u32::try_from(buf.len()).unwrap_or(u32::MAX);

        let fragment = tcp4::FragmentData {
            fragment_length: data_len,
            fragment_buffer: buf.as_mut_ptr().cast::<crate::ffi::c_void>(),
        };
        let mut rx_data = tcp4::ReceiveData {
            urgent_flag: r_efi::efi::Boolean::FALSE,
            data_length: data_len,
            fragment_count: 1,
            fragment_table: [fragment],
        };

        self.read_inner((&raw mut rx_data).cast(), timeout)
    }

    pub(crate) fn read_vectored(
        &self,
        buf: &[IoSliceMut<'_>],
        timeout: Option<Duration>,
    ) -> io::Result<usize> {
        let mut data_length = 0u32;
        let mut fragment_count = 0u32;

        // Calculate how many IoSlice in buf can be transmitted.
        for i in buf {
            // IoSlice length is always <= u32::MAX in UEFI.
            match data_length.checked_add(u32::try_from(i.len()).expect("value is stored as a u32"))
            {
                Some(x) => data_length = x,
                None => break,
            }
            fragment_count += 1;
        }

        let rx_data_size = size_of::<tcp4::ReceiveData<0>>()
            + size_of::<tcp4::FragmentData>() * (fragment_count as usize);
        let mut rx_data = helpers::UefiBox::<tcp4::ReceiveData>::new(rx_data_size)?;
        rx_data.write(tcp4::ReceiveData {
            urgent_flag: r_efi::efi::Boolean::FALSE,
            data_length,
            fragment_count,
            fragment_table: [],
        });
        unsafe {
            // SAFETY: IoSlice and FragmentData are guaranteed to have same layout.
            crate::ptr::copy_nonoverlapping(
                buf.as_ptr().cast(),
                (*rx_data.as_mut_ptr()).fragment_table.as_mut_ptr(),
                fragment_count as usize,
            );
        };

        self.read_inner(rx_data.as_mut_ptr(), timeout)
    }

    pub(crate) fn read_inner(
        &self,
        rx_data: *mut tcp4::ReceiveData,
        timeout: Option<Duration>,
    ) -> io::Result<usize> {
        let evt = unsafe { self.create_evt() }?;
        let completion_token =
            tcp4::CompletionToken { event: evt.as_ptr(), status: Status::SUCCESS };

        let protocol = self.protocol.as_ptr();
        let mut token = tcp4::IoToken { completion_token, packet: tcp4::IoTokenPacket { rx_data } };

        let r = unsafe { ((*protocol).receive)(protocol, &mut token) };
        if r.is_error() {
            return Err(io::Error::from_raw_os_error(r.as_usize()));
        }

        unsafe { self.wait_or_cancel(timeout, &mut token.completion_token) }?;

        if completion_token.status.is_error() {
            Err(io::Error::from_raw_os_error(completion_token.status.as_usize()))
        } else {
            let data_length = unsafe { (*rx_data).data_length };
            Ok(data_length as usize)
        }
    }

    /// Wait for an event to finish. This is checked by an atomic boolean that is supposed to be set
    /// to true in the event callback.
    ///
    /// Optionally, allow specifying a timeout.
    ///
    /// If a timeout is provided, the operation (specified by its `EFI_TCP4_COMPLETION_TOKEN`) is
    /// canceled and Error of kind TimedOut is returned.
    ///
    /// # SAFETY
    ///
    /// Pointer to a valid `EFI_TCP4_COMPLETION_TOKEN`
    unsafe fn wait_or_cancel(
        &self,
        timeout: Option<Duration>,
        token: *mut tcp4::CompletionToken,
    ) -> io::Result<()> {
        if !self.wait_for_flag(timeout) {
            let _ = unsafe { self.cancel(token) };
            return Err(io::Error::new(io::ErrorKind::TimedOut, "Operation Timed out"));
        }

        Ok(())
    }

    /// Abort an asynchronous connection, listen, transmission or receive request.
    ///
    /// If token is NULL, then all pending tokens issued by EFI_TCP4_PROTOCOL.Connect(),
    /// EFI_TCP4_PROTOCOL.Accept(), EFI_TCP4_PROTOCOL.Transmit() or EFI_TCP4_PROTOCOL.Receive() are
    /// aborted.
    ///
    /// # SAFETY
    ///
    /// Pointer to a valid `EFI_TCP4_COMPLETION_TOKEN` or NULL
    unsafe fn cancel(&self, token: *mut tcp4::CompletionToken) -> io::Result<()> {
        let protocol = self.protocol.as_ptr();

        let r = unsafe { ((*protocol).cancel)(protocol, token) };
        if r.is_error() {
            return Err(io::Error::from_raw_os_error(r.as_usize()));
        } else {
            Ok(())
        }
    }

    unsafe fn create_evt(&self) -> io::Result<helpers::OwnedEvent> {
        self.flag.store(false, Ordering::Relaxed);
        helpers::OwnedEvent::new(
            efi::EVT_NOTIFY_SIGNAL,
            efi::TPL_CALLBACK,
            Some(toggle_atomic_flag),
            Some(unsafe { NonNull::new_unchecked(self.flag.as_ptr().cast()) }),
        )
    }

    fn wait_for_flag(&self, timeout: Option<Duration>) -> bool {
        let start = Instant::now();

        while !self.flag.load(Ordering::Relaxed) {
            let _ = self.poll();
            if let Some(t) = timeout {
                if Instant::now().duration_since(start) >= t {
                    return false;
                }
            }
        }

        true
    }

    fn poll(&self) -> io::Result<()> {
        let protocol = self.protocol.as_ptr();
        let r = unsafe { ((*protocol).poll)(protocol) };

        if r.is_error() { Err(io::Error::from_raw_os_error(r.as_usize())) } else { Ok(()) }
    }
}

impl Drop for Tcp4 {
    fn drop(&mut self) {
        let _ = unsafe { self.service_binding.destroy_child(self.handle) };
    }
}

extern "efiapi" fn toggle_atomic_flag(_: r_efi::efi::Event, ctx: *mut crate::ffi::c_void) {
    let flag = unsafe { AtomicBool::from_ptr(ctx.cast()) };
    flag.store(true, Ordering::Relaxed);
}
