use fs::File;
use io::{Error, Result, Read};
use iter::Iterator;
use net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use str::FromStr;
use string::{String, ToString};
use sys::syscall::EINVAL;
use time;
use vec::{IntoIter, Vec};

use self::dns::{Dns, DnsQuery};

pub extern crate libc as netc;
pub use self::tcp::{TcpStream, TcpListener};
pub use self::udp::UdpSocket;

mod dns;
mod tcp;
mod udp;

pub struct LookupHost(IntoIter<SocketAddr>);

impl Iterator for LookupHost {
    type Item = SocketAddr;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

pub fn lookup_host(host: &str) -> Result<LookupHost> {
    let mut ip_string = String::new();
    File::open("/etc/net/ip")?.read_to_string(&mut ip_string)?;
    let ip: Vec<u8> = ip_string.trim().split(".").map(|part| part.parse::<u8>().unwrap_or(0)).collect();

    let mut dns_string = String::new();
    File::open("/etc/net/dns")?.read_to_string(&mut dns_string)?;
    let dns: Vec<u8> = dns_string.trim().split(".").map(|part| part.parse::<u8>().unwrap_or(0)).collect();

    if ip.len() == 4 && dns.len() == 4 {
        let tid = (time::SystemTime::now().duration_since(time::UNIX_EPOCH).unwrap().subsec_nanos() >> 16) as u16;

        let packet = Dns {
            transaction_id: tid,
            flags: 0x0100,
            queries: vec![DnsQuery {
                name: host.to_string(),
                q_type: 0x0001,
                q_class: 0x0001,
            }],
            answers: vec![]
        };

        let packet_data = packet.compile();

        let socket = UdpSocket::bind(&SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::new(ip[0], ip[1], ip[2], ip[3]), 0)))?;
        socket.connect(&SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::new(dns[0], dns[1], dns[2], dns[3]), 53)))?;
        socket.send(&packet_data)?;

        let mut buf = [0; 65536];
        let count = socket.recv(&mut buf)?;

        match Dns::parse(&buf[.. count]) {
            Ok(response) => {
                let mut addrs = vec![];
                for answer in response.answers.iter() {
                    if answer.a_type == 0x0001 && answer.a_class == 0x0001 && answer.data.len() == 4 {
                        addrs.push(SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::new(answer.data[0], answer.data[1], answer.data[2], answer.data[3]), 0)));
                    }
                }
                Ok(LookupHost(addrs.into_iter()))
            },
            Err(_err) => Err(Error::from_raw_os_error(EINVAL))
        }
    } else {
        Err(Error::from_raw_os_error(EINVAL))
    }
}

fn path_to_peer_addr(path_str: &str) -> SocketAddr {
    let mut parts = path_str.split('/').next().unwrap_or("").split(':').skip(1);
    let host = Ipv4Addr::from_str(parts.next().unwrap_or("")).unwrap_or(Ipv4Addr::new(0, 0, 0, 0));
    let port = parts.next().unwrap_or("").parse::<u16>().unwrap_or(0);
    SocketAddr::V4(SocketAddrV4::new(host, port))
}

fn path_to_local_addr(path_str: &str) -> SocketAddr {
    let mut parts = path_str.split('/').nth(1).unwrap_or("").split(':');
    let host = Ipv4Addr::from_str(parts.next().unwrap_or("")).unwrap_or(Ipv4Addr::new(0, 0, 0, 0));
    let port = parts.next().unwrap_or("").parse::<u16>().unwrap_or(0);
    SocketAddr::V4(SocketAddrV4::new(host, port))
}
