// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME
#[allow(missing_doc)];

use fmt;
use io::net::ip::{IpAddr, Ipv4Addr, Ipv6Addr};
use io::{IoResult};
use iter::Iterator;
use libc;
use option::{Option};
use rt::rtio::{IoFactory, LocalIo, RtioRawSocket};
use clone::Clone;
use vec::{Vector, ImmutableVector};

#[cfg(test)]
use vec::MutableVector;

pub struct RawSocket {
    priv obj: ~RtioRawSocket
}

impl RawSocket {
    pub fn new(protocol: Protocol) -> IoResult<RawSocket> {
        LocalIo::maybe_raise(|io| {
            io.raw_socket_new(protocol).map(|s| RawSocket { obj: s })
        })
    }

    pub fn recvfrom(&mut self, buf: &mut [u8]) -> IoResult<(uint, Option<~NetworkAddress>)> {
        self.obj.recvfrom(buf)
    }

    pub fn sendto(&mut self, buf: &[u8], dst: ~NetworkAddress) -> IoResult<int> {
        self.obj.sendto(buf, dst)
    }
}


#[deriving(Clone, Eq, Show)]
pub struct NetworkInterface {
    name: ~str,
    index: u32,
    mac: Option<MacAddr>,
    ipv4: Option<IpAddr>,
    ipv6: Option<IpAddr>,
    flags: u32,
}

impl NetworkInterface {
    pub fn mac_address(&self) -> MacAddr {
        self.mac.unwrap()
    }

    pub fn is_loopback(&self) -> bool {
        self.flags & (libc::IFF_LOOPBACK as u32) != 0
    }
}

#[deriving(Clone, Eq, Show)]
pub enum NetworkAddress {
    IpAddress(IpAddr),
    NetworkAddress(~NetworkInterface)
}

#[deriving(Eq, Clone)]
pub enum MacAddr {
    MacAddr(u8, u8, u8, u8, u8, u8)
}

impl fmt::Show for MacAddr {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            MacAddr(a, b, c, d, e, f) =>
                write!(fmt.buf, "{:x}:{:x}:{:x}:{:x}:{:x}:{:x}",
                       a, b, c, d, e, f)
        }
    }
}

pub trait Packet {
    fn packet<'p>(&'p self) -> &'p [u8];
    fn offset(&self) -> uint;
}

pub struct EthernetHeader<'p> {
    priv packet: &'p [u8],
    priv offset: uint
}

pub struct MutableEthernetHeader<'p> {
    priv packet: &'p mut [u8],
    priv offset: uint
}

impl<'p> Packet for EthernetHeader<'p> {
    #[inline(always)]
    fn packet<'p>(&'p self) -> &'p [u8] { self.packet }
    #[inline(always)]
    fn offset(&self) -> uint { self.offset }
}

impl<'p> Packet for MutableEthernetHeader<'p> {
    #[inline(always)]
    fn packet<'p>(&'p self) -> &'p [u8] { self.packet.as_slice() }
    #[inline(always)]
    fn offset(&self) -> uint { self.offset }
}

pub trait EthernetPacket : Packet {
    fn get_source(&self) -> MacAddr {
        MacAddr(
            self.packet()[self.offset() + 6],
            self.packet()[self.offset() + 7],
            self.packet()[self.offset() + 8],
            self.packet()[self.offset() + 9],
            self.packet()[self.offset() + 10],
            self.packet()[self.offset() + 11]
        )
    }

    fn get_destination(&self) -> MacAddr {
        MacAddr(
            self.packet()[self.offset() + 0],
            self.packet()[self.offset() + 1],
            self.packet()[self.offset() + 2],
            self.packet()[self.offset() + 3],
            self.packet()[self.offset() + 4],
            self.packet()[self.offset() + 5]
        )
    }

    fn get_ethertype(&self) -> u16 {
        (self.packet()[self.offset() + 12] as u16 << 8) | (self.packet()[self.offset() + 13] as u16)
    }
}

impl<'p> EthernetPacket for EthernetHeader<'p> {}
impl<'p> EthernetPacket for MutableEthernetHeader<'p> {}

impl<'p> EthernetHeader<'p> {
    pub fn new(packet: &'p [u8], offset: uint) -> EthernetHeader<'p> {
        EthernetHeader { packet: packet, offset: offset }
    }
}

impl<'p> MutableEthernetHeader<'p> {
    pub fn new(packet: &'p mut [u8], offset: uint) -> MutableEthernetHeader<'p> {
        MutableEthernetHeader { packet: packet, offset: offset }
    }

    pub fn set_source(&mut self, mac: MacAddr) {
        match mac {
            MacAddr(a, b, c, d, e, f) => {
                self.packet[self.offset +  6] = a;
                self.packet[self.offset +  7] = b;
                self.packet[self.offset +  8] = c;
                self.packet[self.offset +  9] = d;
                self.packet[self.offset + 10] = e;
                self.packet[self.offset + 11] = f;
            }
        }
    }

    pub fn set_destination(&mut self, mac: MacAddr) {
        match mac {
            MacAddr(a, b, c, d, e, f) => {
                self.packet[self.offset + 0] = a;
                self.packet[self.offset + 1] = b;
                self.packet[self.offset + 2] = c;
                self.packet[self.offset + 3] = d;
                self.packet[self.offset + 4] = e;
                self.packet[self.offset + 5] = f;
            }
        }
    }

    pub fn set_ethertype(&mut self, ethertype: u16) {
        self.packet[self.offset + 12] = (ethertype >> 8) as u8;
        self.packet[self.offset + 13] = (ethertype & 0xFF) as u8;
    }
}

#[test]
fn ethernet_header_test() {
    let mut packet = [0u8, ..14];
    {
        let mut ethernetHeader = MutableEthernetHeader::new(packet.as_mut_slice(), 0);

        let source = MacAddr(0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc);
        ethernetHeader.set_source(source);
        assert_eq!(ethernetHeader.get_source(), source);

        let dest = MacAddr(0xde, 0xf0, 0x12, 0x34, 0x45, 0x67);
        ethernetHeader.set_destination(dest);
        assert_eq!(ethernetHeader.get_destination(), dest);

        ethernetHeader.set_ethertype(EtherTypes::Ipv6);
        assert_eq!(ethernetHeader.get_ethertype(), EtherTypes::Ipv6);
    }

     let refPacket = [0xde, 0xf0, 0x12, 0x34, 0x45, 0x67, /* destination */
                      0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, /* source */
                      0x86, 0xdd /* ethertype */];
    assert_eq!(refPacket.as_slice(), packet.as_slice());
}

pub struct Ipv4Header<'p> {
    priv packet: &'p [u8],
    priv offset: uint
}

pub struct MutableIpv4Header<'p> {
    priv packet: &'p mut [u8],
    priv offset: uint
}

impl<'p> Packet for Ipv4Header<'p> {
    #[inline(always)]
    fn packet<'p>(&'p self) -> &'p [u8] { self.packet }
    #[inline(always)]
    fn offset(&self) -> uint { self.offset }
}

impl<'p> Packet for MutableIpv4Header<'p> {
    #[inline(always)]
    fn packet<'p>(&'p self) -> &'p [u8] { self.packet.as_slice() }
    #[inline(always)]
    fn offset(&self) -> uint { self.offset }
}

pub trait Ipv4Packet : Packet {
    fn get_version(&self) -> u8 {
        self.packet()[self.offset()] >> 4
    }

    fn get_header_length(&self) -> u8 {
        self.packet()[self.offset()] & 0xF
    }

    fn get_dscp(&self) -> u8 {
        (self.packet()[self.offset() + 1] & 0xFC) >> 2
    }

    fn get_ecn(&self) -> u8 {
        self.packet()[self.offset() + 1] & 3
    }

    fn get_total_length(&self) -> u16 {
        let b1 = self.packet()[self.offset() + 2] as u16 << 8;
        let b2 = self.packet()[self.offset() + 3] as u16;
        b1 | b2
    }

    fn get_identification(&self) -> u16 {
        let b1 = self.packet()[self.offset() + 4] as u16 << 8;
        let b2 = self.packet()[self.offset() + 5] as u16;
        b1 | b2
    }

    fn get_flags(&self) -> u8 {
        self.packet()[self.offset() + 6] >> 5
    }

    fn get_fragment_offset(&self) -> u16 {
        let b1 = (self.packet()[self.offset() + 6] & 0x1F) as u16 << 8;
        let b2 = self.packet()[self.offset() + 7] as u16;
        b1 | b2
    }

    fn get_ttl(&self) -> u8 {
        self.packet()[self.offset() + 8]
    }

    fn get_next_level_protocol(&self) -> IpNextHeaderProtocol {
        self.packet()[self.offset() + 9]
    }

    fn get_checksum(&self) -> u16 {
        let cs1 = self.packet()[self.offset() + 10] as u16 << 8;
        let cs2 = self.packet()[self.offset() + 11] as u16;
        cs1 | cs2
    }

    fn get_source(&self) -> IpAddr {
        Ipv4Addr(self.packet()[self.offset() + 12],
                 self.packet()[self.offset() + 13],
                 self.packet()[self.offset() + 14],
                 self.packet()[self.offset() + 15])
    }

    fn get_destination(&self) -> IpAddr {
        Ipv4Addr(self.packet()[self.offset() + 16],
                 self.packet()[self.offset() + 17],
                 self.packet()[self.offset() + 18],
                 self.packet()[self.offset() + 19])
    }

    fn calculate_checksum(&mut self) -> u16 {
        let len = self.offset() + self.get_header_length() as uint * 4;
        let mut sum = 0u32;
        let mut i = self.offset();
        while i < len {
            let word = self.packet()[i] as u32 << 8 | self.packet()[i + 1] as u32;
            sum = sum + word;
            i = i + 2;
        }
        while sum >> 16 != 0 {
            sum = (sum >> 16) + (sum & 0xFFFF);
        }
        return !sum as u16;
    }
}

impl<'p> Ipv4Packet for Ipv4Header<'p> {}
impl<'p> Ipv4Packet for MutableIpv4Header<'p> {}

impl<'p> Ipv4Header<'p> {
    pub fn new(packet: &'p [u8], offset: uint) -> Ipv4Header<'p> {
        Ipv4Header { packet: packet, offset: offset }
    }
}
impl<'p> MutableIpv4Header<'p> {
    pub fn new(packet: &'p mut [u8], offset: uint) -> MutableIpv4Header<'p> {
        MutableIpv4Header { packet: packet, offset: offset }
    }

    pub fn set_version(&mut self, version: u8) {
        let ver = version << 4;
        self.packet[self.offset] = (self.packet[self.offset] & 0x0F) | ver;
    }

    pub fn set_header_length(&mut self, ihl: u8) {
        let len = ihl & 0xF;
        self.packet[self.offset] = (self.packet[self.offset] & 0xF0) | len;
    }

    pub fn set_dscp(&mut self, dscp: u8) {
        let cp = dscp & 0xFC;
        self.packet[self.offset + 1] = (self.packet[self.offset + 1] & 3) | (cp << 2);
    }

    pub fn set_ecn(&mut self, ecn: u8) {
        let cn = ecn & 3;
        self.packet[self.offset + 1] = (self.packet[self.offset + 1] & 0xFC) | cn;
    }

    pub fn set_total_length(&mut self, len: u16) {
        self.packet[self.offset + 2] = (len >> 8) as u8;
        self.packet[self.offset + 3] = (len & 0xFF) as u8;
    }

    pub fn set_identification(&mut self, identification: u16) {
        self.packet[self.offset + 4] = (identification >> 8) as u8;
        self.packet[self.offset + 5] = (identification & 0x00FF) as u8;
    }

    pub fn set_flags(&mut self, flags: u8) {
        let fs = (flags & 7) << 5;
        self.packet[self.offset + 6] = (self.packet[self.offset + 6] & 0x1F) | fs;
    }

    pub fn set_fragment_offset(&mut self, offset: u16) {
        let fo = offset & 0x1FFF;
        self.packet[self.offset + 6] = (self.packet[self.offset + 6] & 0xE0) |
                                       ((fo & 0xFF00) >> 8) as u8;
        self.packet[self.offset + 7] = (fo & 0xFF) as u8;
    }

    pub fn set_ttl(&mut self, ttl: u8) {
        self.packet[self.offset + 8] = ttl;
    }

    pub fn set_next_level_protocol(&mut self, protocol: IpNextHeaderProtocol) {
        self.packet[self.offset + 9] = protocol;
    }

    pub fn set_checksum(&mut self, checksum: u16) {
        let cs1 = ((checksum & 0xFF00) >> 8) as u8;
        let cs2 = (checksum & 0x00FF) as u8;
        self.packet[self.offset + 10] = cs1;
        self.packet[self.offset + 11] = cs2;
    }

    pub fn set_source(&mut self, ip: IpAddr) {
        match ip {
            Ipv4Addr(a, b, c, d) => {
                self.packet[self.offset + 12] = a;
                self.packet[self.offset + 13] = b;
                self.packet[self.offset + 14] = c;
                self.packet[self.offset + 15] = d;
            },
            _ => ()
        }
    }

    pub fn set_destination(&mut self, ip: IpAddr) {
        match ip {
            Ipv4Addr(a, b, c, d) => {
                self.packet[self.offset + 16] = a;
                self.packet[self.offset + 17] = b;
                self.packet[self.offset + 18] = c;
                self.packet[self.offset + 19] = d;
            },
            _ => ()
        }
    }

    pub fn checksum(&mut self) {
        let checksum = self.calculate_checksum();
        self.set_checksum(checksum);
    }
}

#[test]
fn ipv4_header_test() {
    let mut packet = [0u8, ..20];
    {
        let mut ipHeader = MutableIpv4Header::new(packet.as_mut_slice(), 0);
        ipHeader.set_version(4);
        assert_eq!(ipHeader.get_version(), 4);

        ipHeader.set_header_length(5);
        assert_eq!(ipHeader.get_header_length(), 5);

        ipHeader.set_dscp(4);
        assert_eq!(ipHeader.get_dscp(), 4);

        ipHeader.set_ecn(1);
        assert_eq!(ipHeader.get_ecn(), 1);

        ipHeader.set_total_length(115);
        assert_eq!(ipHeader.get_total_length(), 115);

        ipHeader.set_identification(257);
        assert_eq!(ipHeader.get_identification(), 257);

        ipHeader.set_flags(2);
        assert_eq!(ipHeader.get_flags(), 2);

        ipHeader.set_fragment_offset(257);
        assert_eq!(ipHeader.get_fragment_offset(), 257);

        ipHeader.set_ttl(64);
        assert_eq!(ipHeader.get_ttl(), 64);

        ipHeader.set_next_level_protocol(IpNextHeaderProtocols::Udp);
        assert_eq!(ipHeader.get_next_level_protocol(), IpNextHeaderProtocols::Udp);

        ipHeader.set_source(Ipv4Addr(192, 168, 0, 1));
        assert_eq!(ipHeader.get_source(), Ipv4Addr(192, 168, 0, 1));

        ipHeader.set_destination(Ipv4Addr(192, 168, 0, 199));
        assert_eq!(ipHeader.get_destination(), Ipv4Addr(192, 168, 0, 199));

        ipHeader.checksum();
        assert_eq!(ipHeader.get_checksum(), 0xb64e);
    }

    let refPacket = [0x45,           /* ver/ihl */
                     0x11,           /* dscp/ecn */
                     0x00, 0x73,     /* total len */
                     0x01, 0x01,     /* identification */
                     0x41, 0x01,     /* flags/frag offset */
                     0x40,           /* ttl */
                     0x11,           /* proto */
                     0xb6, 0x4e,     /* checksum */
                     0xc0, 0xa8, 0x00, 0x01, /* source ip */
                     0xc0, 0xa8, 0x00, 0xc7  /* dest ip */];
    assert_eq!(refPacket.as_slice(), packet.as_slice());
}

pub struct Ipv6Header<'p> {
    priv packet: &'p [u8],
    priv offset: uint
}

pub struct MutableIpv6Header<'p> {
    priv packet: &'p mut [u8],
    priv offset: uint
}

impl<'p> Packet for Ipv6Header<'p> {
    #[inline(always)]
    fn packet<'p>(&'p self) -> &'p [u8] { self.packet }
    #[inline(always)]
    fn offset(&self) -> uint { self.offset }
}

impl<'p> Packet for MutableIpv6Header<'p> {
    #[inline(always)]
    fn packet<'p>(&'p self) -> &'p [u8] { self.packet.as_slice() }
    #[inline(always)]
    fn offset(&self) -> uint { self.offset }
}

pub trait Ipv6Packet : Packet {
    fn get_version(&self) -> u8 {
        self.packet()[self.offset()] >> 4
    }

    fn get_traffic_class(&self) -> u8 {
        let tc1 = (self.packet()[self.offset() + 0] & 0x0F) << 4;
        let tc2 = self.packet()[self.offset() + 1] >> 4;
        tc1 | tc2
    }

    fn get_flow_label(&self) -> u32 {
        let fl1 = (self.packet()[self.offset() + 1] as u32 & 0xF) << 16;
        let fl2 =  self.packet()[self.offset() + 2] as u32 << 8;
        let fl3 =  self.packet()[self.offset() + 3] as u32;
        fl1 | fl2 | fl3
    }

    fn get_payload_length(&self) -> u16 {
        let len1 = self.packet()[self.offset() + 4] as u16 << 8;
        let len2 = self.packet()[self.offset() + 5] as u16;
        len1 | len2
    }

    fn get_next_header(&self) -> IpNextHeaderProtocol {
        self.packet()[self.offset() + 6]
    }

    fn get_hop_limit(&self) -> u8 {
        self.packet()[self.offset() + 7]
    }

    fn get_source(&self) -> IpAddr {
        let packet = self.packet();
        let offset = self.offset();

        let a = (packet[offset +  8] as u16 << 8) | packet[offset +  9] as u16;
        let b = (packet[offset + 10] as u16 << 8) | packet[offset + 11] as u16;
        let c = (packet[offset + 12] as u16 << 8) | packet[offset + 13] as u16;
        let d = (packet[offset + 14] as u16 << 8) | packet[offset + 15] as u16;
        let e = (packet[offset + 16] as u16 << 8) | packet[offset + 17] as u16;
        let f = (packet[offset + 18] as u16 << 8) | packet[offset + 19] as u16;
        let g = (packet[offset + 20] as u16 << 8) | packet[offset + 21] as u16;
        let h = (packet[offset + 22] as u16 << 8) | packet[offset + 23] as u16;

        Ipv6Addr(a, b, c, d, e, f, g, h)
    }

    fn get_destination(&self) -> IpAddr {
        let packet = self.packet();
        let offset = self.offset();

        let a = (packet[offset + 24] as u16 << 8) | packet[offset + 25] as u16;
        let b = (packet[offset + 26] as u16 << 8) | packet[offset + 27] as u16;
        let c = (packet[offset + 28] as u16 << 8) | packet[offset + 29] as u16;
        let d = (packet[offset + 30] as u16 << 8) | packet[offset + 31] as u16;
        let e = (packet[offset + 32] as u16 << 8) | packet[offset + 33] as u16;
        let f = (packet[offset + 34] as u16 << 8) | packet[offset + 35] as u16;
        let g = (packet[offset + 36] as u16 << 8) | packet[offset + 37] as u16;
        let h = (packet[offset + 38] as u16 << 8) | packet[offset + 39] as u16;

        Ipv6Addr(a, b, c, d, e, f, g, h)
    }
}

impl<'p> Ipv6Packet for Ipv6Header<'p> {}
impl<'p> Ipv6Packet for MutableIpv6Header<'p> {}

impl<'p> Ipv6Header<'p> {
    pub fn new(packet: &'p [u8], offset: uint) -> Ipv6Header<'p> {
        Ipv6Header { packet: packet, offset: offset }
    }
}

impl<'p> MutableIpv6Header<'p> {
    pub fn new(packet: &'p mut [u8], offset: uint) -> MutableIpv6Header<'p> {
        MutableIpv6Header { packet: packet, offset: offset }
    }

    pub fn set_version(&mut self, version: u8) {
        let ver = version << 4;
        self.packet[self.offset] = (self.packet[self.offset] & 0x0F) | ver;
    }

    pub fn set_traffic_class(&mut self, tc: u8) {
        self.packet[self.offset + 0] = (self.packet[self.offset] & 0xF0) | (tc >> 4);
        self.packet[self.offset + 1] = ((tc & 0x0F) << 4) |
                                        ((self.packet[self.offset + 1] & 0xF0) >> 4);
    }

    pub fn set_flow_label(&mut self, label: u32) {
        let lbl = ((label & 0xF0000) >> 16) as u8;
        self.packet[self.offset + 1] = (self.packet[self.offset + 1] & 0xF0) | lbl;
        self.packet[self.offset + 2] = ((label & 0xFF00) >> 8) as u8;
        self.packet[self.offset + 3] = (label & 0x00FF) as u8;
    }

    pub fn set_payload_length(&mut self, len: u16) {
        self.packet[self.offset + 4] = (len >> 8) as u8;
        self.packet[self.offset + 5] = (len & 0xFF) as u8;
    }

    pub fn set_next_header(&mut self, protocol: IpNextHeaderProtocol) {
        self.packet[self.offset + 6] = protocol;
    }

    pub fn set_hop_limit(&mut self, limit: u8) {
        self.packet[self.offset + 7] = limit;
    }

    pub fn set_source(&mut self, ip: IpAddr) {
        match ip {
            Ipv6Addr(a, b, c, d, e, f, g, h) => {
                self.packet[self.offset +  8] = (a >> 8) as u8;
                self.packet[self.offset +  9] = (a & 0xFF) as u8;
                self.packet[self.offset + 10] = (b >> 8) as u8;
                self.packet[self.offset + 11] = (b & 0xFF) as u8;;
                self.packet[self.offset + 12] = (c >> 8) as u8;
                self.packet[self.offset + 13] = (c & 0xFF) as u8;;
                self.packet[self.offset + 14] = (d >> 8) as u8;
                self.packet[self.offset + 15] = (d & 0xFF) as u8;;
                self.packet[self.offset + 16] = (e >> 8) as u8;
                self.packet[self.offset + 17] = (e & 0xFF) as u8;;
                self.packet[self.offset + 18] = (f >> 8) as u8;
                self.packet[self.offset + 19] = (f & 0xFF) as u8;;
                self.packet[self.offset + 20] = (g >> 8) as u8;
                self.packet[self.offset + 21] = (g & 0xFF) as u8;;
                self.packet[self.offset + 22] = (h >> 8) as u8;
                self.packet[self.offset + 23] = (h & 0xFF) as u8;
            },
            _ => ()
        }
    }

    pub fn set_destination(&mut self, ip: IpAddr) {
        match ip {
            Ipv6Addr(a, b, c, d, e, f, g, h) => {
                self.packet[self.offset + 24] = (a >> 8) as u8;
                self.packet[self.offset + 25] = (a & 0xFF) as u8;
                self.packet[self.offset + 26] = (b >> 8) as u8;
                self.packet[self.offset + 27] = (b & 0xFF) as u8;;
                self.packet[self.offset + 28] = (c >> 8) as u8;
                self.packet[self.offset + 29] = (c & 0xFF) as u8;;
                self.packet[self.offset + 30] = (d >> 8) as u8;
                self.packet[self.offset + 31] = (d & 0xFF) as u8;;
                self.packet[self.offset + 32] = (e >> 8) as u8;
                self.packet[self.offset + 33] = (e & 0xFF) as u8;;
                self.packet[self.offset + 34] = (f >> 8) as u8;
                self.packet[self.offset + 35] = (f & 0xFF) as u8;;
                self.packet[self.offset + 36] = (g >> 8) as u8;
                self.packet[self.offset + 37] = (g & 0xFF) as u8;;
                self.packet[self.offset + 38] = (h >> 8) as u8;
                self.packet[self.offset + 39] = (h & 0xFF) as u8;
            },
            _ => ()
        }
    }
}

#[test]
fn ipv6_header_test() {
    let mut packet = [0u8, ..40];
    {
        let mut ipHeader = MutableIpv6Header::new(packet.as_mut_slice(), 0);
        ipHeader.set_version(6);
        assert_eq!(ipHeader.get_version(), 6);

        ipHeader.set_traffic_class(17);
        assert_eq!(ipHeader.get_traffic_class(), 17);

        ipHeader.set_flow_label(0x10101);
        assert_eq!(ipHeader.get_flow_label(), 0x10101);

        ipHeader.set_payload_length(0x0101);
        assert_eq!(ipHeader.get_payload_length(), 0x0101);

        ipHeader.set_next_header(IpNextHeaderProtocols::Udp);
        assert_eq!(ipHeader.get_next_header(), IpNextHeaderProtocols::Udp);

        ipHeader.set_hop_limit(1);
        assert_eq!(ipHeader.get_hop_limit(), 1)

        let source = Ipv6Addr(0x110, 0x1001, 0x110, 0x1001, 0x110, 0x1001, 0x110, 0x1001);
        ipHeader.set_source(source);
        assert_eq!(ipHeader.get_source(), source);

        let dest = Ipv6Addr(0x110, 0x1001, 0x110, 0x1001, 0x110, 0x1001, 0x110, 0x1001);
        ipHeader.set_destination(dest);
        assert_eq!(ipHeader.get_destination(), dest);
    }

    let refPacket = [0x61,           /* ver/traffic class */
                     0x11,           /* traffic class/flow label */
                     0x01, 0x01,     /* flow label */
                     0x01, 0x01,     /* payload length */
                     0x11,           /* next header */
                     0x01,           /* hop limit */
                     0x01, 0x10,     /* source ip */
                     0x10, 0x01,
                     0x01, 0x10,
                     0x10, 0x01,
                     0x01, 0x10,
                     0x10, 0x01,
                     0x01, 0x10,
                     0x10, 0x01,
                     0x01, 0x10,    /* dest ip */
                     0x10, 0x01,
                     0x01, 0x10,
                     0x10, 0x01,
                     0x01, 0x10,
                     0x10, 0x01,
                     0x01, 0x10,
                     0x10, 0x01];
    assert_eq!(refPacket.as_slice(), packet.as_slice());
}

pub struct UdpHeader<'p> {
    priv packet: &'p [u8],
    priv offset: uint
}
pub struct MutableUdpHeader<'p> {
    priv packet: &'p mut [u8],
    priv offset: uint
}

impl<'p> Packet for UdpHeader<'p> {
    #[inline(always)]
    fn packet<'p>(&'p self) -> &'p [u8] { self.packet }
    #[inline(always)]
    fn offset(&self) -> uint { self.offset }
}

impl<'p> Packet for MutableUdpHeader<'p> {
    #[inline(always)]
    fn packet<'p>(&'p self) -> &'p [u8] { self.packet.as_slice() }
    #[inline(always)]
    fn offset(&self) -> uint { self.offset }
}

pub trait UdpPacket : Packet {
    fn get_source(&self) -> u16 {
        let s1 = self.packet()[self.offset() + 0] as u16 << 8;
        let s2 = self.packet()[self.offset() + 1] as u16;
        s1 | s2
    }

    fn get_destination(&self) -> u16 {
        let d1 = self.packet()[self.offset() + 2] as u16 << 8;
        let d2 = self.packet()[self.offset() + 3] as u16;
        d1 | d2
    }

    fn get_length(&self) -> u16 {
        let l1 = self.packet()[self.offset() + 4] as u16 << 8;
        let l2 = self.packet()[self.offset() + 5] as u16;
        l1 | l2
    }

    fn get_checksum(&self) -> u16 {
        let c1 = self.packet()[self.offset() + 6] as u16 << 8;
        let c2 = self.packet()[self.offset() + 7] as u16;
        c1 | c2
    }

    fn calculate_ipv4_checksum(&self, offset: uint) -> u16 {
        let mut sum = 0u32;
        let mut i = offset;

        // Checksum pseudo-header
        // IPv4 source
        sum = sum + (self.packet()[i + 12] as u32 << 8 | self.packet()[i + 13] as u32);
        sum = sum + (self.packet()[i + 14] as u32 << 8 | self.packet()[i + 15] as u32);

        // IPv4 destination
        sum = sum + (self.packet()[i + 16] as u32 << 8 | self.packet()[i + 17] as u32);
        sum = sum + (self.packet()[i + 18] as u32 << 8 | self.packet()[i + 19] as u32);

        // IPv4 Next level protocol
        sum = sum + self.packet()[i + 9] as u32;

        // UDP Length
        sum = sum + (self.packet()[self.offset() + 4] as u32 << 8 |
                     self.packet()[self.offset() + 5] as u32);

        // Checksum UDP header/packet
        i = self.offset();
        let len = self.offset() + self.get_length() as uint;
        while i < len {
            let word = self.packet()[i] as u32 << 8 | self.packet()[i + 1] as u32;
            sum = sum + word;
            i = i + 2;
        }
        // If the length is odd, make sure to checksum the final byte
        if len & 1 != 0 {
            sum = sum + (self.packet()[len - 1] as u32 << 8);
        }
        while sum >> 16 != 0 {
            sum = (sum >> 16) + (sum & 0xFFFF);
        }

        return !sum as u16;
    }

    fn calculate_ipv6_checksum(&self, offset: uint) -> u16 {
        let mut sum = 0u32;
        let mut i = offset + 8;
        let mut len = offset + 40;

        // Checksum pseudo-header
        // IPv6 source and destination
        while i < len {
            let word = self.packet()[i] as u32 << 8 | self.packet()[i + 1] as u32;
            sum = sum + word;
            i = i + 2;
        }

        // IPv6 Next header
        sum = sum + self.packet()[offset + 6] as u32;

        // UDP Length
        sum = sum + self.get_length() as u32;

        // Checksum UDP header/packet
        i = self.offset();
        len = self.offset() + self.get_length() as uint;
        while i < len {
            let word = self.packet()[i] as u32 << 8 | self.packet()[i + 1] as u32;
            sum = sum + word;
            i = i + 2;
        }
        // If the length is odd, make sure to checksum the final byte
        if len & 1 != 0 {
            sum = sum + self.packet()[len - 1] as u32 << 8;
        }

        while sum >> 16 != 0 {
            sum = (sum >> 16) + (sum & 0xFFFF);
        }

        return !sum as u16;
    }

}

impl<'p> UdpPacket for UdpHeader<'p> {}
impl<'p> UdpPacket for MutableUdpHeader<'p> {}

impl<'p> UdpHeader<'p> {
    pub fn new(packet: &'p [u8], offset: uint) -> UdpHeader<'p> {
        UdpHeader { packet: packet, offset: offset }
    }
}

impl<'p> MutableUdpHeader<'p> {
    pub fn new(packet: &'p mut [u8], offset: uint) -> MutableUdpHeader<'p> {
        MutableUdpHeader { packet: packet, offset: offset }
    }

    pub fn set_source(&mut self, port: u16) {
        self.packet[self.offset + 0] = (port >> 8) as u8;
        self.packet[self.offset + 1] = (port & 0xFF) as u8;
    }

    pub fn set_destination(&mut self, port: u16) {
        self.packet[self.offset + 2] = (port >> 8) as u8;
        self.packet[self.offset + 3] = (port & 0xFF) as u8;
    }

    pub fn set_length(&mut self, len: u16) {
        self.packet[self.offset + 4] = (len >> 8) as u8;
        self.packet[self.offset + 5] = (len & 0xFF) as u8;
    }

    pub fn set_checksum(&mut self, checksum: u16) {
        self.packet[self.offset + 6] = (checksum >> 8) as u8;
        self.packet[self.offset + 7] = (checksum & 0xFF) as u8;
    }

    pub fn ipv4_checksum(&mut self, offset: uint) {
        let checksum = self.calculate_ipv4_checksum(offset);
        // RFC 768, a checksum of zero is transmitted as all ones
        if checksum != 0 {
            self.set_checksum(checksum);
        } else {
            self.set_checksum(0xFFFF);
        }
    }

    pub fn ipv6_checksum(&mut self, offset: uint) {
        let checksum = self.calculate_ipv6_checksum(offset);
        if checksum != 0 {
            self.set_checksum(checksum);
        } else {
            self.set_checksum(0xFFFF);
        }
    }

}

#[test]
fn udp_header_ipv4_test() {
    let mut packet = [0u8, ..20 + 8 + 4];
    {
        let mut ipHeader = MutableIpv4Header::new(packet.as_mut_slice(), 0);
        ipHeader.set_next_level_protocol(IpNextHeaderProtocols::Udp);
        ipHeader.set_source(Ipv4Addr(192, 168, 0, 1));
        ipHeader.set_destination(Ipv4Addr(192, 168, 0, 199));
    }

    // Set data
    packet[20 + 8 + 0] = 't' as u8;
    packet[20 + 8 + 1] = 'e' as u8;
    packet[20 + 8 + 2] = 's' as u8;
    packet[20 + 8 + 3] = 't' as u8;

    {
        let mut udpHeader = MutableUdpHeader::new(packet.as_mut_slice(), 20);
        udpHeader.set_source(12345);
        assert_eq!(udpHeader.get_source(), 12345);

        udpHeader.set_destination(54321);
        assert_eq!(udpHeader.get_destination(), 54321);

        udpHeader.set_length(8 + 4);
        assert_eq!(udpHeader.get_length(), 8 + 4);

        udpHeader.ipv4_checksum(0);
        assert_eq!(udpHeader.get_checksum(), 0x9178);
    }

    let refPacket = [0x30, 0x39, /* source */
                     0xd4, 0x31, /* destination */
                     0x00, 0x0c, /* length */
                     0x91, 0x78  /* checksum*/];
    assert_eq!(refPacket.as_slice(), packet.slice(20, 28));
}


#[test]
fn udp_header_ipv6_test() {
    let mut packet = [0u8, ..40 + 8 + 4];
    {
        let mut ipHeader = MutableIpv6Header::new(packet.as_mut_slice(), 0);
        ipHeader.set_next_header(IpNextHeaderProtocols::Udp);
        ipHeader.set_source(Ipv6Addr(0, 0, 0, 0, 0, 0, 0, 1));
        ipHeader.set_destination(Ipv6Addr(0, 0, 0, 0, 0, 0, 0, 1));
    }

    // Set data
    packet[40 + 8 + 0] = 't' as u8;
    packet[40 + 8 + 1] = 'e' as u8;
    packet[40 + 8 + 2] = 's' as u8;
    packet[40 + 8 + 3] = 't' as u8;

    {
        let mut udpHeader = MutableUdpHeader::new(packet.as_mut_slice(), 20);
        udpHeader.set_source(12345);
        assert_eq!(udpHeader.get_source(), 12345);

        udpHeader.set_destination(54321);
        assert_eq!(udpHeader.get_destination(), 54321);

        udpHeader.set_length(8 + 4);
        assert_eq!(udpHeader.get_length(), 8 + 4);

        udpHeader.ipv6_checksum(0);
        assert_eq!(udpHeader.get_checksum(), 0xf6f3);
    }

    let refPacket = [0x30, 0x39, /* source */
                     0xd4, 0x31, /* destination */
                     0x00, 0x0c, /* length */
                     0xf6, 0xf3  /* checksum*/];
    assert_eq!(refPacket.as_slice(), packet.slice(20, 28));
}


pub enum Protocol {
    DataLinkProtocol(DataLinkProto),
    NetworkProtocol(NetworkProto),
    TransportProtocol(TransportProto)
}

pub enum DataLinkProto {
    EthernetProtocol,
    CookedEthernetProtocol(EtherType)
}

pub enum NetworkProto {
    Ipv4NetworkProtocol,
    Ipv6NetworkProtocol
}

pub enum TransportProto {
    Ipv4TransportProtocol(IpNextHeaderProtocol),
    Ipv6TransportProtocol(IpNextHeaderProtocol)
}

// EtherTypes defined at:
// http://www.iana.org/assignments/ieee-802-numbers/ieee-802-numbers.xhtml
// These values should be used in the Ethernet EtherType field
//
// A handful of these have been selected since most are archaic and unused.
pub mod EtherTypes {
    pub static Ipv4: u16      = 0x0800;
    pub static Arp: u16       = 0x0806;
    pub static WakeOnLan: u16 = 0x0842;
    pub static Rarp: u16      = 0x8035;
    pub static Ipv6: u16      = 0x86DD;
}

pub type EtherType = u16;

// Protocol numbers as defined at:
// http://www.iana.org/assignments/protocol-numbers/protocol-numbers.xhtml
// Above protocol numbers last updated: 2014-01-16
// These values should be used in either the IPv4 Next Level Protocol field
// or the IPv6 Next Header field.
pub mod IpNextHeaderProtocols {
    pub static Hopopt: u8         =   0; // IPv6 Hop-by-Hop Option [RFC2460]
    pub static Icmp: u8           =   1; // Internet Control Message [RFC792]
    pub static Igmp: u8           =   2; // Internet Group Management [RFC1112]
    pub static Ggp: u8            =   3; // Gateway-to-Gateway [RFC823]
    pub static Ipv4: u8           =   4; // IPv4 encapsulation [RFC2003]
    pub static St: u8             =   5; // Stream [RFC1190][RFC1819]
    pub static Tcp: u8            =   6; // Transmission Control [RFC793]
    pub static Cbt: u8            =   7; // CBT
    pub static Egp: u8            =   8; // Exterior Gateway Protocol [RFC888]
    pub static Igp: u8            =   9; // any private interior gateway (used by Cisco for
                                          //                               their IGRP)
    pub static BbnRccMon: u8      =  10; // BBN RCC Monitoring
    pub static NvpII: u8          =  11; // Network Voice Protocol [RFC741]
    pub static Pup: u8            =  12; // PUP
    pub static Argus: u8          =  13; // ARGUS
    pub static Emcon: u8          =  14; // EMCON
    pub static Xnet: u8           =  15; // Cross Net Debugger
    pub static Chaos: u8          =  16; // Chaos
    pub static Udp: u8            =  17; // User Datagram [RFC768]
    pub static Mux: u8            =  18; // Multiplexing
    pub static DcnMeas: u8        =  19; // DCN Measurement Subsystems
    pub static Hmp: u8            =  20; // Host Monitoring [RFC869]
    pub static Prm: u8            =  21; // Packet Radio Measurement
    pub static XnsIdp: u8         =  22; // XEROX NS IDP
    pub static Trunk1: u8         =  23; // Trunk-1
    pub static Trunk2: u8         =  24; // Trunk-2
    pub static Leaf1: u8          =  25; // Leaf-1
    pub static Leaf2: u8          =  26; // Leaf-2
    pub static Rdp: u8            =  27; // Reliable Data Protocol [RFC908]
    pub static Irtp: u8           =  28; // Internet Reliable Transaction [RFC938]
    pub static IsoTp4: u8         =  29; // ISO Transport Protocol Class 4 [RFC905]
    pub static Netblt: u8         =  30; // Bulk Data Transfer Protocol [RFC969]
    pub static MfeNsp: u8         =  31; // MFE Network Services Protocol
    pub static MeritInp: u8       =  32; // MERIT Internodal Protocol
    pub static Dccp: u8           =  33; // Datagram Congestion Control Protocol [RFC4340]
    pub static ThreePc: u8        =  34; // Third Party Connect Protocol
    pub static Idpr: u8           =  35; // Inter-Domain Policy Routing Protocol
    pub static Xtp: u8            =  36; // XTP
    pub static Ddp: u8            =  37; // Datagram Delivery Protocol
    pub static IdprCmtp: u8       =  38; // IDPR Control Message Transport Proto
    pub static TpPlusPlus: u8     =  39; // TP++ Transport Protocol
    pub static Il: u8             =  40; // IL Transport Protocol
    pub static Ipv6: u8           =  41; // IPv6 encapsulation [RFC2473]
    pub static Sdrp: u8           =  42; // Source Demand Routing Protocol
    pub static Ipv6Route: u8      =  43; // Routing Header for IPv6
    pub static Ipv6Frag: u8       =  44; // Fragment Header for IPv6
    pub static Idrp: u8           =  45; // Inter-Domain Routing Protocol
    pub static Rsvp: u8           =  46; // Reservation Protocol [RFC2205][RFC3209]
    pub static Gre: u8            =  47; // Generic Routing Encapsulation [RFC1701]
    pub static Dsr: u8            =  48; // Dynamic Source Routing Protocol [RFC4728]
    pub static Bna: u8            =  49; // BNA
    pub static Esp: u8            =  50; // Encap Security Payload [RFC4303]
    pub static Ah: u8             =  51; // Authentication Header [RFC4302]
    pub static INlsp: u8          =  52; // Integrated Net Layer Security TUBA
    pub static Swipe: u8          =  53; // IP with Encryption
    pub static Narp: u8           =  54; // NBMA Address Resolution Protocol [RFC1735]
    pub static Mobile: u8         =  55; // IP Mobility
    pub static Tlsp: u8           =  56; // Transport Layer Security Protocol using Kryptonet key
                                          // management
    pub static Skip: u8           =  57; // SKIP
    pub static Ipv6Icmp: u8       =  58; // ICMP for IPv6 [RFC2460]
    pub static Ipv6NoNxt: u8      =  59; // No Next Header for IPv6 [RFC2460]
    pub static Ipv6Opts: u8       =  60; // Destination Options for IPv6 [RFC2460]
    pub static HostInternal: u8   =  61; // any host internal protocol
    pub static Cftp: u8           =  62; // CFTP
    pub static LocalNetwork: u8   =  63; // any local network
    pub static SatExpak: u8       =  64; // SATNET and Backroom EXPAK
    pub static Kryptolan: u8      =  65; // Kryptolan
    pub static Rvd: u8            =  66; // MIT Remote Virtual Disk Protocol
    pub static Ippc: u8           =  67; // Internet Pluribus Packet Core
    pub static DistributedFs: u8  =  68; // any distributed file system
    pub static SatMon: u8         =  69; // SATNET Monitoring
    pub static Visa: u8           =  70; // VISA Protocol
    pub static Ipcv: u8           =  71; // Internet Packet Core Utility
    pub static Cpnx: u8           =  72; // Computer Protocol Network Executive
    pub static Cphb: u8           =  73; // Computer Protocol Heart Beat
    pub static Wsn: u8            =  74; // Wang Span Network
    pub static Pvp: u8            =  75; // Packet Video Protocol
    pub static BrSatMon: u8       =  76; // Backroom SATNET Monitoring
    pub static SunNd: u8          =  77; // SUN ND PROTOCOL-Temporary
    pub static WbMon: u8          =  78; // WIDEBAND Monitoring
    pub static WbExpak: u8        =  79; // WIDEBAND EXPAK
    pub static IsoIp: u8          =  80; // ISO Internet Protocol
    pub static Vmtp: u8           =  81; // VMTP
    pub static SecureVmtp: u8     =  82; // SECURE-VMTP
    pub static Vines: u8          =  83; // VINES
    pub static TtpOrIptm: u8      =  84; // Transaction Transport Protocol/IP Traffic Manager
    pub static NsfnetIgp: u8      =  85; // NSFNET-IGP
    pub static Dgp: u8            =  86; // Dissimilar Gateway Protocol
    pub static Tcf: u8            =  87; // TCF
    pub static Eigrp: u8          =  88; // EIGRP
    pub static OspfigP: u8        =  89; // OSPFIGP [RFC1583][RFC2328][RFC5340]
    pub static SpriteRpc: u8      =  90; // Sprite RPC Protocol
    pub static Larp: u8           =  91; // Locus Address Resolution Protocol
    pub static Mtp: u8            =  92; // Multicast Transport Protocol
    pub static Ax25: u8           =  93; // AX.25 Frames
    pub static IpIp: u8           =  94; // IP-within-IP Encapsulation Protocol
    pub static Micp: u8           =  95; // Mobile Internetworking Control Pro.
    pub static SccSp: u8          =  96; // Semaphore Communications Sec. Pro.
    pub static Etherip: u8        =  97; // Ethernet-within-IP Encapsulation [RFC3378]
    pub static Encap: u8          =  98; // Encapsulation Header [RFC1241]
    pub static PrivEncryption: u8 =  99; // any private encryption scheme
    pub static Gmtp: u8           = 100; // GMTP
    pub static Ifmp: u8           = 101; // Ipsilon Flow Management Protocol
    pub static Pnni: u8           = 102; // PNNI over IP
    pub static Pim: u8            = 103; // Protocol Independent Multicast [RFC4601]
    pub static Aris: u8           = 104; // ARIS
    pub static Scps: u8           = 105; // SCPS
    pub static Qnx: u8            = 106; // QNX
    pub static AN: u8             = 107; // Active Networks
    pub static IpComp: u8         = 108; // IP Payload Compression Protocol [RFC2393]
    pub static Snp: u8            = 109; // Sitara Networks Protocol
    pub static CompaqPeer: u8     = 110; // Compaq Peer Protocol
    pub static IpxInIp: u8        = 111; // IPX in IP
    pub static Vrrp: u8           = 112; // Virtual Router Redundancy Protocol [RFC5798]
    pub static Pgm: u8            = 113; // PGM Reliable Transport Protocol
    pub static ZeroHop: u8        = 114; // any 0-hop protocol
    pub static L2tp: u8           = 115; // Layer Two Tunneling Protocol [RFC3931]
    pub static Ddx: u8            = 116; // D-II Data Exchange (DDX)
    pub static Iatp: u8           = 117; // Interactive Agent Transfer Protocol
    pub static Stp: u8            = 118; // Schedule Transfer Protocol
    pub static Srp: u8            = 119; // SpectraLink Radio Protocol
    pub static Uti: u8            = 120; // UTI
    pub static Smp: u8            = 121; // Simple Message Protocol
    pub static Sm: u8             = 122; // Simple Multicast Protocol
    pub static Ptp: u8            = 123; // Performance Transparency Protocol
    pub static IsisOverIpv4: u8   = 124; //
    pub static Fire: u8           = 125; //
    pub static Crtp: u8           = 126; // Combat Radio Transport Protocol
    pub static Crudp: u8          = 127; // Combat Radio User Datagram
    pub static Sscopmce: u8       = 128; //
    pub static Iplt: u8           = 129; //
    pub static Sps: u8            = 130; // Secure Packet Shield
    pub static Pipe: u8           = 131; // Private IP Encapsulation within IP
    pub static Sctp: u8           = 132; // Stream Control Transmission Protocol
    pub static Fc: u8             = 133; // Fibre Channel [RFC6172]
    pub static RsvpE2eIgnore: u8  = 134; // [RFC3175]
    pub static MobilityHeader: u8 = 135; // [RFC6275]
    pub static UdpLite: u8        = 136; // [RFC3828]
    pub static MplsInIp: u8       = 137; // [RFC4023]
    pub static Manet: u8          = 138; // MANET Protocols [RFC5498]
    pub static Hip: u8            = 139; // Host Identity Protocol [RFC5201]
    pub static Shim6: u8          = 140; // Shim6 Protocol [RFC5533]
    pub static Wesp: u8           = 141; // Wrapped Encapsulating Security Payload [RFC5840]
    pub static Rohc: u8           = 142; // Robust Header Compression [RFC5858]
    pub static Test1: u8          = 253; // Use for experimentation and testing [RFC3692]
    pub static Test2: u8          = 254; // Use for experimentation and testing [RFC3692]
    pub static Reserved: u8       = 255; //
}

pub type IpNextHeaderProtocol = u8;

#[cfg(test)]
pub mod test {
    use realstd::clone::Clone;
    use realstd::result::{Ok, Err};
    use realstd::container::Container;
    use realstd::option::{Some};
    use realstd::str::StrSlice;
    use realstd::io::net::raw::*;
    use realstd::io::net::raw::{Ipv4Packet, UdpPacket, IpNextHeaderProtocols};
    use realstd::task::spawn;
    use realstd::io::net::ip::{IpAddr, Ipv4Addr, Ipv6Addr};
    use realstd::vec::ImmutableVector;
    use realstd::iter::Iterator;
    use realstd::vec::{Vector};

    use netsupport;

    // This needs to be redefined here otherwise imports do not work correctly
    macro_rules! iotest (
        { fn $name:ident() $b:block $($a:attr)* } => (
            mod $name {
                #[allow(unused_imports)];

                use realstd::io;
                use realstd::prelude::*;
                use realstd::io::*;
                use realstd::io::fs::*;
                use realstd::io::test::*;
                use realstd::io::net::tcp::*;
                use realstd::io::net::ip::*;
                use realstd::io::net::udp::*;
                use realstd::io::net::raw::*;
                use std::io::net::raw::test::*;
                use std::io::net::raw::EtherTypes;
                #[cfg(unix)]
                use realstd::io::net::unix::*;
                use realstd::io::timer::*;
                use realstd::io::process::*;
                use realstd::unstable::running_on_valgrind;
                use realstd::str;
                use realstd::util;

                fn f() $b

                $($a)* #[test] fn green() { f() }
                $($a)* #[test] fn native() {
                    use native;
                    let (tx, rx) = channel();
                    native::task::spawn(proc() { tx.send(f()) });
                    rx.recv();
                }
            }
        )
    )

    pub static ETHERNET_HEADER_LEN: u16 = 14;
    pub static IPV4_HEADER_LEN: u16 = 20;
    pub static IPV6_HEADER_LEN: u16 = 40;
    pub static UDP_HEADER_LEN: u16 = 8;
    pub static TEST_DATA_LEN: u16 = 4;

    pub fn layer4_test(ip: IpAddr, headerLen: uint) {
        let message = "message";

        spawn( proc() {
            let mut buf: ~[u8] = ~[0, ..128];
            match RawSocket::new(get_proto(ip)) {
                Ok(mut sock) => match sock.recvfrom(buf) {
                    Ok((len, Some(~IpAddress(addr)))) => {
                        assert_eq!(buf.slice(headerLen, message.len()), message.as_bytes());
                        assert_eq!(len, message.len());
                        assert!(addr == ip, "addr != ip");
                    },
                    _ => fail!()
                },
                Err(_) => fail!()
            }
        });

        match RawSocket::new(get_proto(ip)) {
            Ok(mut sock) => match sock.sendto(message.as_bytes(), ~IpAddress(ip)) {
                Ok(res) => assert_eq!(res as uint, message.len()),
                Err(_) => fail!()
            },
            Err(_) => fail!()
        }

        fn get_proto(ip: IpAddr) -> Protocol {
            match ip {
                Ipv4Addr(..) =>
                    TransportProtocol(Ipv4TransportProtocol(IpNextHeaderProtocols::Test1)),
                Ipv6Addr(..) =>
                    TransportProtocol(Ipv6TransportProtocol(IpNextHeaderProtocols::Test1))
            }
        }
    }

    iotest!(fn layer4_ipv4() {
        layer4_test(Ipv4Addr(127, 0, 0, 1), IPV4_HEADER_LEN as uint);
    } #[cfg(hasroot)])

    iotest!(fn layer4_ipv6() {
        layer4_test(Ipv6Addr(0, 0, 0, 0, 0, 0, 0, 1), IPV6_HEADER_LEN as uint);
    } #[cfg(hasroot)])

    pub fn build_ipv4_header(packet: &mut [u8], offset: uint) {
        let mut ipHeader = MutableIpv4Header::new(packet, offset);

        ipHeader.set_version(4);
        ipHeader.set_header_length(5);
        ipHeader.set_total_length(IPV4_HEADER_LEN + UDP_HEADER_LEN + TEST_DATA_LEN);
        ipHeader.set_ttl(4);
        ipHeader.set_next_level_protocol(IpNextHeaderProtocols::Udp);
        ipHeader.set_source(Ipv4Addr(127, 0, 0, 1));
        ipHeader.set_destination(Ipv4Addr(127, 0, 0, 1));
        ipHeader.checksum();
    }

    pub fn build_ipv6_header(packet: &mut [u8], offset: uint) {
        let mut ipHeader = MutableIpv6Header::new(packet, offset);

        ipHeader.set_version(6);
        ipHeader.set_payload_length(UDP_HEADER_LEN + TEST_DATA_LEN);
        ipHeader.set_next_header(IpNextHeaderProtocols::Udp);
        ipHeader.set_hop_limit(4);
        ipHeader.set_source(Ipv6Addr(0, 0, 0, 0, 0, 0, 0, 1));
        ipHeader.set_destination(Ipv6Addr(0, 0, 0, 0, 0, 0, 0, 1));
    }

    pub fn build_udp_header(packet: &mut [u8], offset: uint) {
        let mut udpHeader = MutableUdpHeader::new(packet, offset);

        udpHeader.set_source(1234); // Arbitary port number
        udpHeader.set_destination(1234);
        udpHeader.set_length(UDP_HEADER_LEN + TEST_DATA_LEN);
    }

    pub fn build_udp4_packet(packet: &mut [u8], start: uint) {
        build_ipv4_header(packet, start);
        build_udp_header(packet, IPV4_HEADER_LEN as uint);

        let dataStart = IPV4_HEADER_LEN + UDP_HEADER_LEN;
        packet[dataStart + 0] = 't' as u8;
        packet[dataStart + 1] = 'e' as u8;
        packet[dataStart + 2] = 's' as u8;
        packet[dataStart + 3] = 't' as u8;

        MutableUdpHeader::new(packet, IPV4_HEADER_LEN as uint).ipv4_checksum(start);
    }

    pub fn build_udp6_packet(packet: &mut [u8], start: uint) {
        build_ipv6_header(packet, start);
        build_udp_header(packet, IPV6_HEADER_LEN as uint);

        let dataStart = IPV6_HEADER_LEN + UDP_HEADER_LEN;
        packet[dataStart + 0] = 't' as u8;
        packet[dataStart + 1] = 'e' as u8;
        packet[dataStart + 2] = 's' as u8;
        packet[dataStart + 3] = 't' as u8;

        MutableUdpHeader::new(packet, IPV6_HEADER_LEN as uint).ipv6_checksum(start);
    }

    pub fn get_test_interface() -> NetworkInterface {
        (**netsupport::get_network_interfaces()
            .as_slice().iter()
            .filter(|x| x.is_loopback())
            .next()
            .unwrap())
            .clone()
    }

    pub fn same_ports(packet1: &[u8], packet2: &[u8], offset: uint) -> bool {
        {
            let ip1 = Ipv4Header::new(packet1, offset);
            let ip2 = Ipv4Header::new(packet2, offset);

            // Check we have an IPv4/UDP packet
            if ip1.get_version() != 4 || ip2.get_version() != 4 ||
               ip1.get_next_level_protocol() != IpNextHeaderProtocols::Udp ||
               ip2.get_next_level_protocol() != IpNextHeaderProtocols::Udp {
                return false;
            }
        }

        let udp1 = UdpHeader::new(packet1, offset + IPV4_HEADER_LEN as uint);
        let udp2 = UdpHeader::new(packet2, offset + IPV4_HEADER_LEN as uint);

        if udp1.get_source() == udp2.get_source() &&
           udp1.get_destination() == udp2.get_destination() {
            return true;
        }

        return false;
    }

    iotest!(fn layer3_ipv4_test() {
        let sendAddr = Ipv4Addr(127, 0, 0, 1);
        let mut packet = [0u8, ..IPV4_HEADER_LEN + UDP_HEADER_LEN + TEST_DATA_LEN];
        build_udp4_packet(packet.as_mut_slice(), 0);

        spawn( proc() {
            let mut buf: ~[u8] = ~[0, ..128];
            match RawSocket::new(NetworkProtocol(Ipv4NetworkProtocol)) {
                Ok(mut sock) => match sock.recvfrom(buf) {
                    Ok((len, Some(~IpAddress(addr)))) => {
                        assert_eq!(buf.slice(0, packet.len()), packet.as_slice());
                        assert_eq!(len, packet.len());
                        assert!(addr == sendAddr, "addr != sendAddr");
                    },
                    _ => fail!()
                },
                Err(_) => fail!()
            }
        });

        match RawSocket::new(NetworkProtocol(Ipv4NetworkProtocol)) {
            Ok(mut sock) => match sock.sendto(packet, ~IpAddress(sendAddr)) {
                Ok(res) => assert_eq!(res as uint, packet.len()),
                Err(_) => fail!()
            },
            Err(_) => fail!()
        }

    } #[cfg(hasroot)])

    iotest!(fn layer3_ipv6_test() {
        let sendAddr = Ipv6Addr(0, 0, 0, 0, 0, 0, 0, 1);
        let mut packet = [0u8, ..IPV6_HEADER_LEN + UDP_HEADER_LEN + TEST_DATA_LEN];
        build_udp6_packet(packet.as_mut_slice(), 0);

        spawn( proc() {
            let mut buf: ~[u8] = ~[0, ..128];
            match RawSocket::new(NetworkProtocol(Ipv6NetworkProtocol)) {
                Ok(mut sock) => match sock.recvfrom(buf) {
                    Ok((len, Some(~IpAddress(addr)))) => {
                        assert_eq!(buf.slice(0, packet.len()), packet.as_slice());
                        assert_eq!(len, packet.len());
                        assert!(addr == sendAddr, "addr != sendAddr");
                    },
                    _ => fail!()
                },
                Err(_) => fail!()
            }
        });

        match RawSocket::new(NetworkProtocol(Ipv6NetworkProtocol)) {
            Ok(mut sock) => match sock.sendto(packet, ~IpAddress(sendAddr)) {
                Ok(res) => assert_eq!(res as uint, packet.len()),
                Err(_) => fail!()
            },
            Err(_) => fail!()
        }

    } #[cfg(hasroot)])

    iotest!(fn layer2_cooked_test() {
        let interface = get_test_interface();
        let interface2 = get_test_interface();

        let mut packet = [0u8, ..32];

        build_udp4_packet(packet.as_mut_slice(), 0);

        let (tx, rx) = channel();
        let (tx2, rx2) = channel();

        spawn( proc() {
            let mut buf: ~[u8] = ~[0, ..128];
            match RawSocket::new(DataLinkProtocol(CookedEthernetProtocol(EtherTypes::Ipv4))) {
                Ok(mut sock) => {
                    tx.send(());
                    loop {
                        match sock.recvfrom(buf) {
                            Ok((len, Some(~NetworkAddress(ni)))) => {
                                if len == packet.len() && same_ports(packet.as_slice(), buf, 0) {
                                    assert_eq!(buf.slice(0, packet.len()), packet.as_slice());
                                    assert!(*ni == interface, "*ni != interface");
                                    break;
                                }
                            },
                            _ => fail!()
                        }
                    }
                },
                Err(_) => fail!()
            }
            rx2.recv();
        });

        match RawSocket::new(DataLinkProtocol(CookedEthernetProtocol(EtherTypes::Ipv4))) {
            Ok(mut sock) => {
                rx.recv();
                match sock.sendto(packet, ~NetworkAddress(~interface2)) {
                    Ok(res) => assert_eq!(res as uint, packet.len()),
                    Err(_) => fail!()
                }
            },
            Err(_) => fail!()
        }
        tx2.send(());
    } #[cfg(hasroot)] #[ignore(cfg(not(target_os = "linux")))])

    iotest!(fn layer2_test() {
        let interface = get_test_interface();
        let interface2 = interface.clone();

        let mut packet = [0u8, ..46];

        {
            let mut ethernetHeader = MutableEthernetHeader::new(packet.as_mut_slice(), 0);
            ethernetHeader.set_source(interface.mac_address());
            ethernetHeader.set_destination(interface.mac_address());
            ethernetHeader.set_ethertype(EtherTypes::Ipv4);
        }

        build_udp4_packet(packet.as_mut_slice(), ETHERNET_HEADER_LEN as uint);

        let (tx, rx) = channel();
        let (tx2, rx2) = channel();

        spawn( proc() {
            let mut buf: ~[u8] = ~[0, ..128];
            match RawSocket::new(DataLinkProtocol(EthernetProtocol)) {
                Ok(mut sock) => {
                    tx.send(());
                    loop {
                        match sock.recvfrom(buf) {
                            Ok((len, Some(~NetworkAddress(ni)))) => {
                                if len == packet.len() &&
                                  same_ports(packet.as_slice(), buf, ETHERNET_HEADER_LEN as uint) {
                                    assert_eq!(buf.slice(0, packet.len()), packet.as_slice());
                                    assert!(*ni == interface, "*ni != interface");
                                    break;
                                }
                            },
                            _ => fail!()
                        }
                    }
                },
                Err(_) => fail!()
            }
            rx2.recv();
        });

        match RawSocket::new(DataLinkProtocol(EthernetProtocol)) {
            Ok(mut sock) => {
                rx.recv();
                match sock.sendto(packet, ~NetworkAddress(~interface2)) {
                    Ok(res) => assert_eq!(res as uint, packet.len()),
                    Err(_) => fail!()
                }
            },
            Err(_) => fail!()
        }
        tx2.send(());
    } #[cfg(hasroot)] #[ignore(cfg(not(target_os = "linux")))])

}
