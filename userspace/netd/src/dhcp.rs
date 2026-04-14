//! DHCPv4 client using smoltcp.
extern crate alloc;
use alloc::string::ToString;
use core::default::Default;

use smoltcp::iface::{Interface, SocketSet, SocketStorage};
use smoltcp::phy::Device;
use smoltcp::socket::dhcpv4::{Event, Socket as Dhcpv4Socket};
use smoltcp::time::{Duration, Instant};
use smoltcp::wire::{IpCidr, Ipv4Address};

#[derive(Debug)]
#[allow(dead_code)]
pub enum DhcpError {
    Timeout,
    Failed,
}

pub struct DhcpConfig {
    pub ip: Ipv4Address,
    pub prefix_len: u8,
    pub gateway: Ipv4Address,
    pub dns: Ipv4Address,
}

fn now() -> Instant {
    Instant::from_millis(stem::time::now().as_millis() as i64)
}

pub fn run_dhcp<D: Device>(iface: &mut Interface, device: &mut D) -> Result<DhcpConfig, DhcpError> {
    let mut sockets_storage: [SocketStorage; 1] = Default::default();
    let mut socket_set = SocketSet::new(&mut sockets_storage[..]);
    let dhcp_handle = socket_set.add(Dhcpv4Socket::new());

    stem::info!("DHCP: Starting discovery...");

    let start = now();
    let timeout = start + Duration::from_secs(30);

    loop {
        let ts = now();
        if ts > timeout {
            return Err(DhcpError::Timeout);
        }

        let _ = iface.poll(ts, device, &mut socket_set);

        let dhcp_socket = socket_set.get_mut::<Dhcpv4Socket>(dhcp_handle);
        if let Some(event) = dhcp_socket.poll() {
            match event {
                Event::Configured(config) => {
                    stem::info!("DHCP: Configuration received");

                    let ip = config.address.address();
                    let gateway = config.router.unwrap_or(Ipv4Address::UNSPECIFIED);
                    let dns = config
                        .dns_servers
                        .first()
                        .copied()
                        .unwrap_or(Ipv4Address::UNSPECIFIED);
                    let prefix_len = config.address.prefix_len();

                    iface.update_ip_addrs(|addrs| {
                        addrs.clear();
                        let _ = addrs.push(IpCidr::Ipv4(config.address));
                    });

                    if let Some(route) = config.router {
                        let _ = iface.routes_mut().add_default_ipv4_route(route);
                    }

                    return Ok(DhcpConfig {
                        ip,
                        prefix_len,
                        gateway,
                        dns,
                    });
                }
                Event::Deconfigured => {
                    stem::warn!("DHCP: Deconfigured");
                }
            }
        }

        let delay = iface.poll_delay(ts, &socket_set);
        let wait_ms = delay.map(|d| d.total_millis()).unwrap_or(100).min(100);
        stem::time::sleep_ms(wait_ms as u64);
    }
}
