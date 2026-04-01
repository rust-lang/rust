use crate::io;
use crate::net::{Ipv4Addr, SocketAddr, SocketAddrV4, SocketAddrV6};
use crate::os::xous::ffi::lend_mut;
use crate::os::xous::services::{DnsLendMut, dns_server};

#[repr(C, align(4096))]
struct LookupHostQuery([u8; 4096]);

pub struct LookupHost {
    data: LookupHostQuery,
    port: u16,
    offset: usize,
    count: usize,
}

impl Iterator for LookupHost {
    type Item = SocketAddr;
    fn next(&mut self) -> Option<SocketAddr> {
        if self.offset >= self.data.0.len() {
            return None;
        }
        match self.data.0.get(self.offset) {
            Some(&4) => {
                self.offset += 1;
                if self.offset + 4 > self.data.0.len() {
                    return None;
                }
                let result = Some(SocketAddr::V4(SocketAddrV4::new(
                    Ipv4Addr::new(
                        self.data.0[self.offset],
                        self.data.0[self.offset + 1],
                        self.data.0[self.offset + 2],
                        self.data.0[self.offset + 3],
                    ),
                    self.port,
                )));
                self.offset += 4;
                result
            }
            Some(&6) => {
                self.offset += 1;
                if self.offset + 16 > self.data.0.len() {
                    return None;
                }
                let mut new_addr = [0u8; 16];
                for (src, octet) in self.data.0[(self.offset + 1)..(self.offset + 16 + 1)]
                    .iter()
                    .zip(new_addr.iter_mut())
                {
                    *octet = *src;
                }
                let result =
                    Some(SocketAddr::V6(SocketAddrV6::new(new_addr.into(), self.port, 0, 0)));
                self.offset += 16;
                result
            }
            _ => None,
        }
    }
}

pub fn lookup_host(query: &str, port: u16) -> io::Result<LookupHost> {
    let mut result = LookupHost { data: LookupHostQuery([0u8; 4096]), offset: 0, count: 0, port };

    // Copy the query into the message that gets sent to the DNS server
    for (query_byte, result_byte) in query.as_bytes().iter().zip(result.data.0.iter_mut()) {
        *result_byte = *query_byte;
    }

    lend_mut(
        dns_server(),
        DnsLendMut::RawLookup.into(),
        &mut result.data.0,
        0,
        query.as_bytes().len(),
    )
    .unwrap();
    if result.data.0[0] != 0 {
        return Err(io::const_error!(io::ErrorKind::InvalidInput, "DNS failure"));
    }
    assert_eq!(result.offset, 0);
    result.count = result.data.0[1] as usize;

    // Advance the offset to the first record
    result.offset = 2;
    Ok(result)
}
