use r_efi::efi::{self, Status};
use r_efi::protocols::tcp4;

use crate::io;
use crate::net::SocketAddrV4;
use crate::ptr::NonNull;
use crate::sync::atomic::{AtomicBool, Ordering};
use crate::sys::pal::helpers;

const TYPE_OF_SERVICE: u8 = 8;
const TIME_TO_LIVE: u8 = 255;

pub(crate) struct Tcp4 {
    protocol: NonNull<tcp4::Protocol>,
    flag: AtomicBool,
    #[expect(dead_code)]
    service_binding: helpers::ServiceProtocol,
}

const DEFAULT_ADDR: efi::Ipv4Address = efi::Ipv4Address { addr: [0u8; 4] };

impl Tcp4 {
    pub(crate) fn new() -> io::Result<Self> {
        let service_binding = helpers::ServiceProtocol::open(tcp4::SERVICE_BINDING_PROTOCOL_GUID)?;
        let protocol = helpers::open_protocol(service_binding.child_handle(), tcp4::PROTOCOL_GUID)?;

        Ok(Self { service_binding, protocol, flag: AtomicBool::new(false) })
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

        // FIXME: Remove when passive connections with proper subnet handling are added
        assert!(station_address.is_none());
        let use_default_address = efi::Boolean::TRUE;
        let (station_address, station_port) = (DEFAULT_ADDR, 0);
        let subnet_mask = helpers::ipv4_to_r_efi(crate::net::Ipv4Addr::new(0, 0, 0, 0));

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
            control_option: crate::ptr::null_mut(),
        };

        let r = unsafe { ((*protocol).configure)(protocol, &mut config_data) };
        if r.is_error() { Err(crate::io::Error::from_raw_os_error(r.as_usize())) } else { Ok(()) }
    }

    pub(crate) fn connect(&self) -> io::Result<()> {
        let evt = unsafe { self.create_evt() }?;
        let completion_token =
            tcp4::CompletionToken { event: evt.as_ptr(), status: Status::SUCCESS };

        let protocol = self.protocol.as_ptr();
        let mut conn_token = tcp4::ConnectionToken { completion_token };

        let r = unsafe { ((*protocol).connect)(protocol, &mut conn_token) };
        if r.is_error() {
            return Err(io::Error::from_raw_os_error(r.as_usize()));
        }

        self.wait_for_flag();

        if completion_token.status.is_error() {
            Err(io::Error::from_raw_os_error(completion_token.status.as_usize()))
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

    fn wait_for_flag(&self) {
        while !self.flag.load(Ordering::Relaxed) {
            let _ = self.poll();
        }
    }

    fn poll(&self) -> io::Result<()> {
        let protocol = self.protocol.as_ptr();
        let r = unsafe { ((*protocol).poll)(protocol) };

        if r.is_error() { Err(io::Error::from_raw_os_error(r.as_usize())) } else { Ok(()) }
    }
}

extern "efiapi" fn toggle_atomic_flag(_: r_efi::efi::Event, ctx: *mut crate::ffi::c_void) {
    let flag = unsafe { AtomicBool::from_ptr(ctx.cast()) };
    flag.store(true, Ordering::Relaxed);
}
