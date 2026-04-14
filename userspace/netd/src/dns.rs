//! Simple DNS client for A record lookups.
extern crate alloc;
use alloc::string::ToString;
use core::default::Default;

use alloc::vec::Vec;
use smoltcp::iface::{Interface, SocketSet, SocketStorage};
use smoltcp::phy::Device;
use smoltcp::socket::udp::{self, PacketMetadata, Socket as UdpSocket};
use smoltcp::time::{Duration, Instant};
use smoltcp::wire::{IpAddress, IpEndpoint, Ipv4Address};

fn now() -> Instant {
    Instant::from_millis(stem::time::now().as_millis() as i64)
}

#[derive(Debug)]
pub enum DnsError {
    Timeout,
    InvalidResponse,
    NoAnswer,
}

pub fn lookup_a<D: Device>(
    iface: &mut Interface,
    device: &mut D,
    dns_server: Ipv4Address,
    name: &str,
) -> Result<Ipv4Address, DnsError> {
    let mut rx_meta = [PacketMetadata::EMPTY; 4];
    let mut rx_data = [0u8; 2048];
    let mut tx_meta = [PacketMetadata::EMPTY; 4];
    let mut tx_data = [0u8; 2048];

    let udp_rx_buffer = udp::PacketBuffer::new(&mut rx_meta[..], &mut rx_data[..]);
    let udp_tx_buffer = udp::PacketBuffer::new(&mut tx_meta[..], &mut tx_data[..]);
    let mut udp_socket = UdpSocket::new(udp_rx_buffer, udp_tx_buffer);

    let local_port = 49152u16;
    if let Err(e) = udp_socket.bind(local_port) {
        stem::warn!("DNS: Failed to bind socket to port {}: {:?}", local_port, e);
        return Err(DnsError::Timeout);
    }

    let mut sockets_storage: [SocketStorage; 1] = Default::default();
    let mut socket_set = SocketSet::new(&mut sockets_storage[..]);
    let udp_handle = socket_set.add(udp_socket);

    let query = build_dns_query(name);
    let endpoint = IpEndpoint::new(IpAddress::Ipv4(dns_server), 53);

    stem::info!("DNS: Querying {} for {}", dns_server, name);

    let start = now();
    let timeout = start + Duration::from_secs(5);
    let mut sent = false;
    let mut poll_count = 0u32;

    loop {
        let ts = now();
        if ts > timeout {
            stem::info!("DNS: Timeout after {} polls", poll_count);
            return Err(DnsError::Timeout);
        }

        let _ = iface.poll(ts, device, &mut socket_set);
        poll_count += 1;

        let socket = socket_set.get_mut::<UdpSocket>(udp_handle);
        if !sent && socket.can_send() {
            let _ = socket.send_slice(&query, endpoint);
            sent = true;
            stem::info!(
                "DNS: Query sent to {}:53 (txid=0x1234, {} bytes)",
                dns_server,
                query.len()
            );
        }

        if socket.can_recv() {
            let (data, _) = socket.recv().map_err(|_| DnsError::InvalidResponse)?;
            stem::info!("DNS: Response received ({} bytes)", data.len());
            return parse_dns_response(data);
        }

        stem::time::sleep_ms(10);
    }
}

fn build_dns_query(name: &str) -> Vec<u8> {
    let mut query = Vec::new();
    query.extend_from_slice(&[
        0x12, 0x34, // txid
        0x01, 0x00, // standard query
        0x00, 0x01, // qdcount
        0x00, 0x00, // ancount
        0x00, 0x00, // nscount
        0x00, 0x00, // arcount
    ]);

    for part in name.split('.') {
        query.push(part.len() as u8);
        query.extend_from_slice(part.as_bytes());
    }
    query.push(0);
    query.extend_from_slice(&[
        0x00, 0x01, // A
        0x00, 0x01, // IN
    ]);
    query
}

fn parse_dns_response(data: &[u8]) -> Result<Ipv4Address, DnsError> {
    if data.len() < 12 {
        return Err(DnsError::InvalidResponse);
    }

    let ancount = u16::from_be_bytes([data[6], data[7]]);
    if ancount == 0 {
        return Err(DnsError::NoAnswer);
    }

    let mut pos = 12;
    while pos < data.len() && data[pos] != 0 {
        let len = data[pos] as usize;
        if len >= 192 {
            pos += 2;
            break;
        }
        pos += 1 + len;
    }
    if pos < data.len() && data[pos] == 0 {
        pos += 1;
    }

    pos += 4;

    for _ in 0..ancount {
        if pos >= data.len() {
            return Err(DnsError::InvalidResponse);
        }

        if data[pos] >= 192 {
            pos += 2;
        } else {
            while pos < data.len() && data[pos] != 0 {
                let len = data[pos] as usize;
                pos += 1 + len;
            }
            pos += 1;
        }

        if pos + 10 > data.len() {
            return Err(DnsError::InvalidResponse);
        }

        let rtype = u16::from_be_bytes([data[pos], data[pos + 1]]);
        let rdlength = u16::from_be_bytes([data[pos + 8], data[pos + 9]]);
        pos += 10;

        if rtype == 1 && rdlength == 4 {
            if pos + 4 > data.len() {
                return Err(DnsError::InvalidResponse);
            }
            return Ok(Ipv4Address::from_bytes(&data[pos..pos + 4]));
        }

        pos += rdlength as usize;
    }

    Err(DnsError::NoAnswer)
}
