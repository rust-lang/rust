use crate::fs::File;
use crate::io::{Error, Read, self};
use crate::iter::Iterator;
use crate::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use crate::str::FromStr;
use crate::string::{String, ToString};
use crate::sys::syscall::EINVAL;
use crate::time::{self, Duration};
use crate::vec::{IntoIter, Vec};
use crate::convert::{TryFrom, TryInto};

use self::dns::{Dns, DnsQuery};

pub use self::tcp::{TcpStream, TcpListener};
pub use self::udp::UdpSocket;

pub mod netc;

mod dns;
mod tcp;
mod udp;

pub struct LookupHost(IntoIter<SocketAddr>, u16);

impl LookupHost {
    pub fn port(&self) -> u16 {
        self.1
    }
}

impl Iterator for LookupHost {
    type Item = SocketAddr;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl TryFrom<&str> for LookupHost {
    type Error = io::Error;

    fn try_from(s: &str) -> io::Result<LookupHost> {
        macro_rules! try_opt {
            ($e:expr, $msg:expr) => (
                match $e {
                    Some(r) => r,
                    None => return Err(io::Error::new(io::ErrorKind::InvalidInput,
                                                      $msg)),
                }
            )
        }

        // split the string by ':' and convert the second part to u16
        let mut parts_iter = s.rsplitn(2, ':');
        let port_str = try_opt!(parts_iter.next(), "invalid socket address");
        let host = try_opt!(parts_iter.next(), "invalid socket address");
        let port: u16 = try_opt!(port_str.parse().ok(), "invalid port value");

        (host, port).try_into()
    }
}

impl<'a> TryFrom<(&'a str, u16)> for LookupHost {
    type Error = io::Error;

    fn try_from((host, port): (&'a str, u16)) -> io::Result<LookupHost> {
        let mut ip_string = String::new();
        File::open("/etc/net/ip")?.read_to_string(&mut ip_string)?;
        let ip: Vec<u8> = ip_string.trim().split('.').map(|part| part.parse::<u8>()
                                   .unwrap_or(0)).collect();

        let mut dns_string = String::new();
        File::open("/etc/net/dns")?.read_to_string(&mut dns_string)?;
        let dns: Vec<u8> = dns_string.trim().split('.').map(|part| part.parse::<u8>()
                                     .unwrap_or(0)).collect();

        if ip.len() == 4 && dns.len() == 4 {
            let time = time::SystemTime::now().duration_since(time::UNIX_EPOCH).unwrap();
            let tid = (time.subsec_nanos() >> 16) as u16;

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

            let my_ip = Ipv4Addr::new(ip[0], ip[1], ip[2], ip[3]);
            let dns_ip = Ipv4Addr::new(dns[0], dns[1], dns[2], dns[3]);
            let socket = UdpSocket::bind(Ok(&SocketAddr::V4(SocketAddrV4::new(my_ip, 0))))?;
            socket.set_read_timeout(Some(Duration::new(5, 0)))?;
            socket.set_write_timeout(Some(Duration::new(5, 0)))?;
            socket.connect(Ok(&SocketAddr::V4(SocketAddrV4::new(dns_ip, 53))))?;
            socket.send(&packet_data)?;

            let mut buf = [0; 65536];
            let count = socket.recv(&mut buf)?;

            match Dns::parse(&buf[.. count]) {
                Ok(response) => {
                    let mut addrs = vec![];
                    for answer in response.answers.iter() {
                        if answer.a_type == 0x0001 && answer.a_class == 0x0001
                           && answer.data.len() == 4
                        {
                            let answer_ip = Ipv4Addr::new(answer.data[0],
                                                          answer.data[1],
                                                          answer.data[2],
                                                          answer.data[3]);
                            addrs.push(SocketAddr::V4(SocketAddrV4::new(answer_ip, 0)));
                        }
                    }
                    Ok(LookupHost(addrs.into_iter(), port))
                },
                Err(_err) => Err(Error::from_raw_os_error(EINVAL))
            }
        } else {
            Err(Error::from_raw_os_error(EINVAL))
        }
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
