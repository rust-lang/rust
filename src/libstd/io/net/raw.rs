// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//use clone::Clone;
//use cast;
use io::net::ip::{IpAddr, Ipv4Addr, Ipv6Addr};
use io::{IoResult};
use iter::Iterator;
use num;
use option::{Option, Some};
use rt::rtio::{IoFactory, LocalIo, RtioRawSocket};
use vec::{MutableVector, ImmutableVector};

pub struct RawSocket {
    priv obj: ~RtioRawSocket
}

impl RawSocket {
    pub fn new(protocol: Protocol) -> IoResult<RawSocket> {
        LocalIo::maybe_raise(|io| {
            io.raw_socket_new(protocol).map(|s| RawSocket { obj: s })
        })
    }

    pub fn get_interfaces() -> ~[NetworkInterface] {
        ~[] // FIXME
    }

    pub fn recvfrom(&mut self, buf: &mut [u8]) -> IoResult<(uint, Option<NetworkAddress>)> {
        self.obj.recvfrom(buf)
    }

    pub fn sendto(&mut self, buf: &[u8], dst: Option<NetworkAddress>) -> IoResult<int> {
        self.obj.sendto(buf, dst)
    }
}

pub struct NetworkInterface;


impl NetworkInterface {
    pub fn mac_address(&self) -> MacAddr {
        MacAddr(0, 0, 0, 0, 0, 0) // FIXME
    }

    pub fn is_loopback(&self) -> bool {
        false // FIXME
    }
}

pub struct EthernetHeader<'p> {
    priv packet: &'p mut [u8],
    priv offset: uint
}

impl<'p> EthernetHeader<'p> {
    pub fn new(packet: &'p mut [u8], offset: uint) -> EthernetHeader<'p> {
        EthernetHeader { packet: packet, offset: offset }
    }

    pub fn set_source(&mut self, _mac: MacAddr) {
        // FIXME
    }

    pub fn get_source(&self) -> MacAddr {
        // FIXME
        MacAddr(0, 0, 0, 0, 0, 0)
    }

    pub fn set_destination(&mut self, _mac: MacAddr) {
        // FIXME
    }

    pub fn get_destination(&self) -> MacAddr {
        // FIXME
        MacAddr(0, 0, 0, 0, 0, 0)
    }

    pub fn set_ethertype(&mut self, _ethertype: u16) {
        // FIXME
    }

    pub fn get_ethertype(&self) -> u16 {
        // FIXME
        0
    }
}

pub struct Ipv4Header<'p> {
    priv packet: &'p mut [u8],
    priv offset: uint
}

impl<'p> Ipv4Header<'p> {
    pub fn new(packet: &'p mut [u8], offset: uint) -> Ipv4Header<'p> {
        Ipv4Header { packet: packet, offset: offset }
    }

    pub fn set_version(&mut self, version: u8) {
        let ver = version << 4;
        self.packet[self.offset] = (self.packet[self.offset] & 0x0F) | ver;
    }

    pub fn get_version(&self) -> u8 {
        self.packet[self.offset] >> 4
    }

    pub fn set_header_length(&mut self, ihl: u8) {
        let len = ihl & 0xF;
        self.packet[self.offset] = (self.packet[self.offset] & 0xF0) | len;
    }

    pub fn get_header_length(&self) -> u8 {
        self.packet[self.offset] & 0xF
    }

    pub fn set_dscp(&mut self, dscp: u8) {
        let cp = dscp & 0xFC;
        self.packet[self.offset + 1] = (self.packet[self.offset + 1] & 3) | cp;
    }

    pub fn get_dscp(&self) -> u8 {
        self.packet[self.offset + 1] & 0xFC
    }

    pub fn set_ecn(&mut self, ecn: u8) {
        let cn = ecn & 3;
        self.packet[self.offset + 1] = (self.packet[self.offset + 1] & 0xFC) | cn;
    }

    pub fn get_ecn(&self) -> u8 {
        self.packet[self.offset + 1] & 3
    }

    pub fn set_total_length(&mut self, len: u16) {
        self.packet[self.offset + 2] = (len >> 8) as u8;
        self.packet[self.offset + 3] = (len & 0xFF) as u8;
    }

    pub fn get_total_length(&self) -> u16 {
        let b1 = self.packet[self.offset + 2] as u16 << 8;
        let b2 = self.packet[self.offset + 3] as u16;
        b1 | b2
    }

    pub fn set_identification(&mut self, identification: u16) {
        self.packet[self.offset + 4] = (identification >> 8) as u8;
        self.packet[self.offset + 5] = (identification & 0x00FF) as u8;
    }

    pub fn get_identification(&self) -> u16 {
        let b1 = self.packet[self.offset + 4] as u16 << 8;
        let b2 = self.packet[self.offset + 5] as u16;
        b1 | b2
    }

    pub fn set_flags(&mut self, flags: u8) {
        let fs = (flags & 7) << 5;
        self.packet[self.offset + 6] = (self.packet[self.offset + 6] & 0x1F) | fs;
    }

    pub fn get_flags(&self) -> u8 {
        self.packet[self.offset + 6] >> 5
    }

    pub fn set_fragment_offset(&mut self, offset: u16) {
        let fo = offset & 0x1FFF;
        self.packet[self.offset + 6] = (self.packet[self.offset + 6] & 0xE0) | (fo & 0xFF00) as u8;
        self.packet[self.offset + 7] = (fo & 0xFF) as u8;
    }

    pub fn get_fragment_offset(&self) -> u16 {
        let b1 = (self.packet[self.offset + 6] & 0x1F) as u16 << 8;
        let b2 = self.packet[self.offset + 7] as u16;
        b1 | b2
    }

    pub fn set_ttl(&mut self, ttl: u8) {
        self.packet[self.offset + 8] = ttl;
    }

    pub fn get_ttl(&self) -> u8 {
        self.packet[self.offset + 8]
    }

    pub fn set_next_level_protocol(&mut self, protocol: IpNextHeaderProtocol) {
        self.packet[self.offset + 9] = protocol as u8;
    }

    pub fn get_next_level_protocol(&self) -> IpNextHeaderProtocol {
        num::FromPrimitive::from_u8(self.packet[self.offset + 9]).unwrap()
    }

    pub fn set_checksum(&mut self, checksum: u16) {
        let cs1 = ((checksum & 0xFF00) >> 8) as u8;
        let cs2 = (checksum & 0x00FF) as u8;
        self.packet[self.offset + 10] = cs1;
        self.packet[self.offset + 11] = cs2;
    }

    pub fn get_checksum(&self) -> u16 {
        let cs1 = self.packet[self.offset + 10] as u16 << 8;
        let cs2 = self.packet[self.offset + 11] as u16;
        cs1 | cs2
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

    pub fn get_source(&self) -> IpAddr {
        Ipv4Addr(self.packet[self.offset + 12],
                 self.packet[self.offset + 13],
                 self.packet[self.offset + 14],
                 self.packet[self.offset + 15])
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

    pub fn get_destination(&self) -> IpAddr {
        Ipv4Addr(self.packet[self.offset + 16],
                 self.packet[self.offset + 17],
                 self.packet[self.offset + 18],
                 self.packet[self.offset + 19])
    }

    pub fn checksum(&mut self) {
        let len = self.offset + self.get_header_length() as uint * 4;
        let mut sum = 0u32;
        let mut i = self.offset;
        while i < len {
            let word = self.packet[i] as u32 << 8 | self.packet[i + 1] as u32;
            sum = sum + word;
            i = i + 2;
        }
        while sum >> 16 != 0 {
            sum = (sum >> 16) + (sum & 0xFFFF);
        }
        self.set_checksum(!sum as u16);
    }
}

#[test]
fn ipv4_header_test() {
    let mut packet = [0u8, ..20];
    {
        let mut ipHeader = Ipv4Header::new(packet.as_mut_slice(), 0);
        ipHeader.set_version(4);
        assert_eq!(ipHeader.get_version(), 4);

        ipHeader.set_header_length(5);
        assert_eq!(ipHeader.get_header_length(), 5);

        ipHeader.set_total_length(115);
        assert_eq!(ipHeader.get_total_length(), 115);

        ipHeader.set_flags(2);
        assert_eq!(ipHeader.get_flags(), 2);

        ipHeader.set_ttl(64);
        assert_eq!(ipHeader.get_ttl(), 64);

        ipHeader.set_next_level_protocol(IpNextHeaderProtocol::Udp);
        assert_eq!(ipHeader.get_next_level_protocol(), IpNextHeaderProtocol::Udp);

        ipHeader.set_source(Ipv4Addr(192, 168, 0, 1));
        assert_eq!(ipHeader.get_source(), Ipv4Addr(192, 168, 0, 1));

        ipHeader.set_destination(Ipv4Addr(192, 168, 0, 199));
        assert_eq!(ipHeader.get_destination(), Ipv4Addr(192, 168, 0, 199));

        ipHeader.checksum();
        assert_eq!(ipHeader.get_checksum(), 0xb861);
    }

    let refPacket = [0x45,           /* ver/ihl */
                     0x00,           /* dscp/ecn */
                     0x00, 0x73,     /* total len */
                     0x00, 0x00,     /* identification */
                     0x40, 0x00,     /* flags/frag offset */
                     0x40,           /* ttl */
                     0x11,           /* proto */
                     0xb8, 0x61,     /* checksum */
                     0xc0, 0xa8, 0x00, 0x01, /* source ip */
                     0xc0, 0xa8, 0x00, 0xc7  /* dest ip */];
    assert_eq!(packet, refPacket);
}

pub struct Ipv6Header<'p> {
    priv packet: &'p mut [u8],
    priv offset: uint
}

// FIXME Support extension headers
impl<'p> Ipv6Header<'p> {
    pub fn new(packet: &'p mut [u8], offset: uint) -> Ipv6Header<'p> {
        Ipv6Header { packet: packet, offset: offset }
    }

    pub fn set_version(&mut self, _version: u8) {
        // FIXME
    }

    pub fn get_version(&self) -> uint {
        // FIXME
        0
    }

    pub fn set_traffic_class(&mut self, _tc: u8) {
        // FIXME
    }

    pub fn get_traffic_class(&self) -> u8 {
        // FIXME
        0
    }

    pub fn set_flow_label(&mut self, _label: u32) {
        // FIXME
    }

    pub fn get_flow_label(&self) -> u32 {
        // FIXME
        0
    }

    pub fn set_payload_length(&mut self, _len: u16) {
        // FIXME
    }

    pub fn get_payload_length(&self) -> u16 {
        // FIXME
        0
    }

    pub fn set_next_header(&mut self, _protocol: IpNextHeaderProtocol) {
        // FIXME
    }

    pub fn get_next_header(&self) -> IpNextHeaderProtocol {
        // FIXME
        IpNextHeaderProtocol::Tcp
    }

    pub fn set_hop_limit(&mut self, _limit: u8) {
        // FIXME
    }

    pub fn get_hop_limit(&self) -> u8 {
        // FIXME
        0
    }

    pub fn set_source(&mut self, _ip: IpAddr) {
        // FIXME
    }

    pub fn get_source(&self) -> IpAddr {
        // FIXME
        Ipv6Addr(0, 0, 0, 0, 0, 0, 0, 0)
    }

    pub fn set_destination(&mut self, _ip: IpAddr) {
        // FIXME
    }

    pub fn get_destination(&self) -> IpAddr {
        // FIXME
        Ipv6Addr(0, 0, 0, 0, 0, 0, 0, 0)
    }
}

pub struct UdpHeader<'p> {
    priv packet: &'p mut [u8],
    priv offset: uint
}

impl<'p> UdpHeader<'p> {
    pub fn new(packet: &'p mut [u8], offset: uint) -> UdpHeader<'p> {
        UdpHeader { packet: packet, offset: offset }
    }

    pub fn set_source(&mut self, port: u16) {
        self.packet[self.offset + 0] = (port >> 8) as u8;
        self.packet[self.offset + 1] = (port & 0xFF) as u8;
    }

    pub fn get_source(&self) -> u16 {
        let s1 = self.packet[self.offset + 0] as u16 << 8;
        let s2 = self.packet[self.offset + 1] as u16;
        s1 | s2
    }

    pub fn set_destination(&mut self, port: u16) {
        self.packet[self.offset + 2] = (port >> 8) as u8;
        self.packet[self.offset + 3] = (port & 0xFF) as u8;
    }

    pub fn get_destination(&self) -> u16 {
        let d1 = self.packet[self.offset + 2] as u16 << 8;
        let d2 = self.packet[self.offset + 3] as u16;
        d1 | d2
    }

    pub fn set_length(&mut self, len: u16) {
        self.packet[self.offset + 4] = (len >> 8) as u8;
        self.packet[self.offset + 5] = (len & 0xFF) as u8;
    }

    pub fn get_length(&self) -> u16 {
        let l1 = self.packet[self.offset + 4] as u16 << 8;
        let l2 = self.packet[self.offset + 5] as u16;
        l1 | l2
    }

    pub fn set_checksum(&mut self, checksum: u16) {
        self.packet[self.offset + 6] = (checksum >> 8) as u8;
        self.packet[self.offset + 7] = (checksum & 0xFF) as u8;
    }

    pub fn get_checksum(&self) -> u16 {
        let c1 = self.packet[self.offset + 6] as u16 << 8;
        let c2 = self.packet[self.offset + 7] as u16;
        c1 | c2
    }

    pub fn checksum(&mut self) {
        // FIXME
    }
}

#[deriving(Eq)]
pub enum NetworkAddress {
    IpAddress(IpAddr),
    MacAddress(MacAddr)
}

#[deriving(Eq)]
pub enum MacAddr {
    MacAddr(u8, u8, u8, u8, u8, u8)
}

pub enum Protocol {
    DataLinkProtocol(DataLinkProto),
    NetworkProtocol(NetworkProto),
    TransportProtocol(TransportProto)
}

pub enum DataLinkProto {
    EthernetProtocol(NetworkInterface),
    CookedEthernetProtocol(NetworkInterface)
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
pub static Ipv4EtherType: u16      = 0x0800;
pub static ArpEtherType: u16       = 0x0806;
pub static WakeOnLanEtherType: u16 = 0x0842;
pub static RarpEtherType: u16      = 0x8035;
pub static Ipv6EtherType: u16      = 0x86DD;

// Protocol numbers as defined at:
// http://www.iana.org/assignments/protocol-numbers/protocol-numbers.xhtml
// Above protocol numbers last updated: 2014-01-16
// These values should be used in either the IPv4 Next Level Protocol field
// or the IPv6 Next Header field.
pub mod IpNextHeaderProtocol {
    //use num::FromPrimitive;
    #[deriving(Eq,FromPrimitive)]
    pub enum IpNextHeaderProtocol {
        Hopopt         =   0, // IPv6 Hop-by-Hop Option [RFC2460]
        Icmp           =   1, // Internet Control Message [RFC792]
        Igmp           =   2, // Internet Group Management [RFC1112]
        Ggp            =   3, // Gateway-to-Gateway [RFC823]
        Ipv4           =   4, // IPv4 encapsulation [RFC2003]
        St             =   5, // Stream [RFC1190][RFC1819]
        Tcp            =   6, // Transmission Control [RFC793]
        Cbt            =   7, // CBT
        Egp            =   8, // Exterior Gateway Protocol [RFC888]
        Igp            =   9, // any private interior gateway (used by Cisco for their IGRP)
        BbnRccMon      =  10, // BBN RCC Monitoring
        NvpII          =  11, // Network Voice Protocol [RFC741]
        Pup            =  12, // PUP
        Argus          =  13, // ARGUS
        Emcon          =  14, // EMCON
        Xnet           =  15, // Cross Net Debugger
        Chaos          =  16, // Chaos
        Udp            =  17, // User Datagram [RFC768]
        Mux            =  18, // Multiplexing
        DcnMeas        =  19, // DCN Measurement Subsystems
        Hmp            =  20, // Host Monitoring [RFC869]
        Prm            =  21, // Packet Radio Measurement
        XnsIdp         =  22, // XEROX NS IDP
        Trunk1         =  23, // Trunk-1
        Trunk2         =  24, // Trunk-2
        Leaf1          =  25, // Leaf-1
        Leaf2          =  26, // Leaf-2
        Rdp            =  27, // Reliable Data Protocol [RFC908]
        Irtp           =  28, // Internet Reliable Transaction [RFC938]
        IsoTp4         =  29, // ISO Transport Protocol Class 4 [RFC905]
        Netblt         =  30, // Bulk Data Transfer Protocol [RFC969]
        MfeNsp         =  31, // MFE Network Services Protocol
        MeritInp       =  32, // MERIT Internodal Protocol
        Dccp           =  33, // Datagram Congestion Control Protocol [RFC4340]
        ThreePc        =  34, // Third Party Connect Protocol
        Idpr           =  35, // Inter-Domain Policy Routing Protocol
        Xtp            =  36, // XTP
        Ddp            =  37, // Datagram Delivery Protocol
        IdprCmtp       =  38, // IDPR Control Message Transport Proto
        TpPlusPlus     =  39, // TP++ Transport Protocol
        Il             =  40, // IL Transport Protocol
        Ipv6           =  41, // IPv6 encapsulation [RFC2473]
        Sdrp           =  42, // Source Demand Routing Protocol
        Ipv6Route      =  43, // Routing Header for IPv6
        Ipv6Frag       =  44, // Fragment Header for IPv6
        Idrp           =  45, // Inter-Domain Routing Protocol
        Rsvp           =  46, // Reservation Protocol [RFC2205][RFC3209]
        Gre            =  47, // Generic Routing Encapsulation [RFC1701]
        Dsr            =  48, // Dynamic Source Routing Protocol [RFC4728]
        Bna            =  49, // BNA
        Esp            =  50, // Encap Security Payload [RFC4303]
        Ah             =  51, // Authentication Header [RFC4302]
        INlsp          =  52, // Integrated Net Layer Security TUBA
        Swipe          =  53, // IP with Encryption
        Narp           =  54, // NBMA Address Resolution Protocol [RFC1735]
        Mobile         =  55, // IP Mobility
        Tlsp           =  56, // Transport Layer Security Protocol using Kryptonet key management
        Skip           =  57, // SKIP
        Ipv6Icmp       =  58, // ICMP for IPv6 [RFC2460]
        Ipv6NoNxt      =  59, // No Next Header for IPv6 [RFC2460]
        Ipv6Opts       =  60, // Destination Options for IPv6 [RFC2460]
        HostInternal   =  61, // any host internal protocol
        Cftp           =  62, // CFTP
        LocalNetwork   =  63, // any local network
        SatExpak       =  64, // SATNET and Backroom EXPAK
        Kryptolan      =  65, // Kryptolan
        Rvd            =  66, // MIT Remote Virtual Disk Protocol
        Ippc           =  67, // Internet Pluribus Packet Core
        DistributedFs  =  68, // any distributed file system
        SatMon         =  69, // SATNET Monitoring
        Visa           =  70, // VISA Protocol
        Ipcv           =  71, // Internet Packet Core Utility
        Cpnx           =  72, // Computer Protocol Network Executive
        Cphb           =  73, // Computer Protocol Heart Beat
        Wsn            =  74, // Wang Span Network
        Pvp            =  75, // Packet Video Protocol
        BrSatMon       =  76, // Backroom SATNET Monitoring
        SunNd          =  77, // SUN ND PROTOCOL-Temporary
        WbMon          =  78, // WIDEBAND Monitoring
        WbExpak        =  79, // WIDEBAND EXPAK
        IsoIp          =  80, // ISO Internet Protocol
        Vmtp           =  81, // VMTP
        SecureVmtp     =  82, // SECURE-VMTP
        Vines          =  83, // VINES
        TtpOrIptm      =  84, // Transaction Transport Protocol/Internet Protocol Traffic Manager
        NsfnetIgp      =  85, // NSFNET-IGP
        Dgp            =  86, // Dissimilar Gateway Protocol
        Tcf            =  87, // TCF
        Eigrp          =  88, // EIGRP
        OspfigP        =  89, // OSPFIGP [RFC1583][RFC2328][RFC5340]
        SpriteRpc      =  90, // Sprite RPC Protocol
        Larp           =  91, // Locus Address Resolution Protocol
        Mtp            =  92, // Multicast Transport Protocol
        Ax25           =  93, // AX.25 Frames
        IpIp           =  94, // IP-within-IP Encapsulation Protocol
        Micp           =  95, // Mobile Internetworking Control Pro.
        SccSp          =  96, // Semaphore Communications Sec. Pro.
        Etherip        =  97, // Ethernet-within-IP Encapsulation [RFC3378]
        Encap          =  98, // Encapsulation Header [RFC1241]
        PrivEncryption =  99, // any private encryption scheme
        Gmtp           = 100, // GMTP
        Ifmp           = 101, // Ipsilon Flow Management Protocol
        Pnni           = 102, // PNNI over IP
        Pim            = 103, // Protocol Independent Multicast [RFC4601]
        Aris           = 104, // ARIS
        Scps           = 105, // SCPS
        Qnx            = 106, // QNX
        AN             = 107, // Active Networks
        IpComp         = 108, // IP Payload Compression Protocol [RFC2393]
        Snp            = 109, // Sitara Networks Protocol
        CompaqPeer     = 110, // Compaq Peer Protocol
        IpxInIp        = 111, // IPX in IP
        Vrrp           = 112, // Virtual Router Redundancy Protocol [RFC5798]
        Pgm            = 113, // PGM Reliable Transport Protocol
        ZeroHop        = 114, // any 0-hop protocol
        L2tp           = 115, // Layer Two Tunneling Protocol [RFC3931]
        Ddx            = 116, // D-II Data Exchange (DDX)
        Iatp           = 117, // Interactive Agent Transfer Protocol
        Stp            = 118, // Schedule Transfer Protocol
        Srp            = 119, // SpectraLink Radio Protocol
        Uti            = 120, // UTI
        Smp            = 121, // Simple Message Protocol
        Sm             = 122, // Simple Multicast Protocol
        Ptp            = 123, // Performance Transparency Protocol
        IsisOverIpv4   = 124, //
        Fire           = 125, //
        Crtp           = 126, // Combat Radio Transport Protocol
        Crudp          = 127, // Combat Radio User Datagram
        Sscopmce       = 128, //
        Iplt           = 129, //
        Sps            = 130, // Secure Packet Shield
        Pipe           = 131, // Private IP Encapsulation within IP
        Sctp           = 132, // Stream Control Transmission Protocol
        Fc             = 133, // Fibre Channel [RFC6172]
        RsvpE2eIgnore  = 134, // [RFC3175]
        MobilityHeader = 135, // [RFC6275]
        UdpLite        = 136, // [RFC3828]
        MplsInIp       = 137, // [RFC4023]
        Manet          = 138, // MANET Protocols [RFC5498]
        Hip            = 139, // Host Identity Protocol [RFC5201]
        Shim6          = 140, // Shim6 Protocol [RFC5533]
        Wesp           = 141, // Wrapped Encapsulating Security Payload [RFC5840]
        Rohc           = 142, // Robust Header Compression [RFC5858]
        Test1          = 253, // Use for experimentation and testing [RFC3692]
        Test2          = 254, // Use for experimentation and testing [RFC3692]
        Reserved       = 255, //
    }
}

pub type IpNextHeaderProtocol = self::IpNextHeaderProtocol::IpNextHeaderProtocol;

#[cfg(test)]
pub mod test {
    use result::{Ok, Err};
    use iter::Iterator;
    use container::Container;
    use option::{Some};
    use str::StrSlice;
    use super::*;
    use task::spawn;
    use io::net::ip::{IpAddr, Ipv4Addr, Ipv6Addr};
    use vec::ImmutableVector;

    pub static ETHERNET_HEADER_LEN: u16 = 14;
    pub static IPV4_HEADER_LEN: u16 = 20;
    pub static IPV6_HEADER_LEN: u16 = 40;
    pub static UDP_HEADER_LEN: u16 = 8;
    pub static TEST_DATA_LEN: u16 = 4;

    pub fn layer4_test(ip: IpAddr, headerLen: uint) {
        let message = "message";
        let proto = match ip {
            Ipv4Addr(..) => TransportProtocol(Ipv4TransportProtocol(IpNextHeaderProtocol::Test1)),
            Ipv6Addr(..) => TransportProtocol(Ipv6TransportProtocol(IpNextHeaderProtocol::Test1))
        };
        spawn( proc() {
            let mut buf: ~[u8] = ~[0, ..128];
            match RawSocket::new(proto) {
                Ok(mut sock) => match sock.recvfrom(buf) {
                    Ok((len, Some(IpAddress(addr)))) => {
                        assert_eq!(buf.slice(headerLen, message.len()), message.as_bytes());
                        assert_eq!(len, message.len());
                        assert_eq!(addr, ip);
                    },
                    _ => fail!()
                },
                Err(_) => fail!()
            }
        });

        match RawSocket::new(proto) {
            Ok(mut sock) => match sock.sendto(message.as_bytes(), Some(IpAddress(ip))) {
                Ok(res) => assert_eq!(res as uint, message.len()),
                Err(_) => fail!()
            },
            Err(_) => fail!()
        }
    }

    iotest!(fn layer4_ipv4() {
        layer4_test(Ipv4Addr(127, 0, 0, 1), IPV4_HEADER_LEN as uint);
    } #[cfg(hasroot)])

    iotest!(fn layer4_ipv6() {
        layer4_test(Ipv6Addr(0, 0, 0, 0, 0, 0, 0, 1), IPV6_HEADER_LEN as uint);
    } #[cfg(hasroot)])

    pub fn build_ipv4_header(packet: &mut [u8], offset: uint) {
        let mut ipHeader = Ipv4Header::new(packet, offset);

        ipHeader.set_version(4);
        ipHeader.set_header_length(5);
        ipHeader.set_total_length(IPV4_HEADER_LEN + UDP_HEADER_LEN + TEST_DATA_LEN);
        ipHeader.set_ttl(4);
        ipHeader.set_next_level_protocol(IpNextHeaderProtocol::Udp);
        ipHeader.set_source(Ipv4Addr(127, 0, 0, 1));
        ipHeader.set_destination(Ipv4Addr(127, 0, 0, 1));
        ipHeader.checksum();
    }

    pub fn build_ipv6_header(packet: &mut [u8], offset: uint) {
        let mut ipHeader = Ipv6Header::new(packet, offset);

        ipHeader.set_version(6);
        ipHeader.set_payload_length(UDP_HEADER_LEN + TEST_DATA_LEN);
        ipHeader.set_next_header(IpNextHeaderProtocol::Udp);
        ipHeader.set_hop_limit(4);
        ipHeader.set_source(Ipv6Addr(0, 0, 0, 0, 0, 0, 0, 1));
        ipHeader.set_destination(Ipv6Addr(0, 0, 0, 0, 0, 0, 0, 1));
    }

    pub fn build_udp_header(packet: &mut [u8], offset: uint) {
        let mut udpHeader = UdpHeader::new(packet, offset);

        udpHeader.set_source(1234); // Arbitary port number
        udpHeader.set_destination(1234);
        udpHeader.set_length(UDP_HEADER_LEN + TEST_DATA_LEN);
        udpHeader.checksum();
    }

    pub fn build_udp4_packet(packet: &mut [u8], start: uint) {
        build_ipv4_header(packet, start);
        build_udp_header(packet, IPV4_HEADER_LEN as uint);

        let dataStart = IPV4_HEADER_LEN + UDP_HEADER_LEN;
        packet[dataStart + 0] = 't' as u8;
        packet[dataStart + 1] = 'e' as u8;
        packet[dataStart + 2] = 's' as u8;
        packet[dataStart + 3] = 't' as u8;
    }

    pub fn build_udp6_packet(packet: &mut [u8], start: uint) {
        build_ipv6_header(packet, start);
        build_udp_header(packet, IPV6_HEADER_LEN as uint);

        let dataStart = IPV6_HEADER_LEN + UDP_HEADER_LEN;
        packet[dataStart + 0] = 't' as u8;
        packet[dataStart + 1] = 'e' as u8;
        packet[dataStart + 2] = 's' as u8;
        packet[dataStart + 3] = 't' as u8;
    }

    pub fn get_test_interface() -> NetworkInterface {
        *RawSocket::get_interfaces()
            .iter()
            .filter(|&x| x.is_loopback())
            .next()
            .unwrap()
    }

    iotest!(fn layer3_ipv4_test() {
        let sendAddr = Ipv4Addr(127, 0, 0, 1);
        let mut packet = [0u8, ..IPV4_HEADER_LEN + UDP_HEADER_LEN + TEST_DATA_LEN];
        build_udp4_packet(packet.as_mut_slice(), 0);

        spawn( proc() {
            let mut buf: ~[u8] = ~[0, ..128];
            match RawSocket::new(NetworkProtocol(Ipv4NetworkProtocol)) {
                Ok(mut sock) => match sock.recvfrom(buf) {
                    Ok((len, Some(IpAddress(addr)))) => {
                        assert_eq!(buf.slice(0, packet.len()), packet.as_slice());
                        assert_eq!(len, packet.len());
                        assert_eq!(addr, sendAddr);
                    },
                    _ => fail!()
                },
                Err(_) => fail!()
            }
        });

        match RawSocket::new(NetworkProtocol(Ipv4NetworkProtocol)) {
            Ok(mut sock) => match sock.sendto(packet, Some(IpAddress(sendAddr))) {
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
                    Ok((len, Some(IpAddress(addr)))) => {
                        assert_eq!(buf.slice(0, packet.len()), packet.as_slice());
                        assert_eq!(len, packet.len());
                        assert_eq!(addr, sendAddr);
                    },
                    _ => fail!()
                },
                Err(_) => fail!()
            }
        });

        match RawSocket::new(NetworkProtocol(Ipv6NetworkProtocol)) {
            Ok(mut sock) => match sock.sendto(packet, Some(IpAddress(sendAddr))) {
                Ok(res) => assert_eq!(res as uint, packet.len()),
                Err(_) => fail!()
            },
            Err(_) => fail!()
        }

    } #[cfg(hasroot)])

    iotest!(fn layer2_cooked_test() {
        let interface = get_test_interface();

        let mut packet = [0u8, ..32];

        build_udp4_packet(packet.as_mut_slice(), 0);

        spawn( proc() {
            let mut buf: ~[u8] = ~[0, ..128];
            match RawSocket::new(DataLinkProtocol(CookedEthernetProtocol(interface))) {
                Ok(mut sock) => match sock.recvfrom(buf) {
                    Ok((len, Some(MacAddress(addr)))) => {
                        assert_eq!(buf.slice(0, packet.len()), packet.as_slice());
                        assert_eq!(len, packet.len());
                        assert_eq!(addr, interface.mac_address());
                    },
                    _ => fail!()
                },
                Err(_) => fail!()
            }
        });

        match RawSocket::new(DataLinkProtocol(CookedEthernetProtocol(interface))) {
            Ok(mut sock) => match sock.sendto(packet, Some(MacAddress(interface.mac_address()))) {
                Ok(res) => assert_eq!(res as uint, packet.len()),
                Err(_) => fail!()
            },
            Err(_) => fail!()
        }
    } #[cfg(hasroot)])

    iotest!(fn layer2_test() {
        let interface = get_test_interface();

        let mut packet = [0u8, ..46];

        {
            let mut ethernetHeader = EthernetHeader::new(packet.as_mut_slice(), 0);
            ethernetHeader.set_source(interface.mac_address());
            ethernetHeader.set_destination(interface.mac_address());
            ethernetHeader.set_ethertype(Ipv4EtherType);
        }

        build_udp4_packet(packet.as_mut_slice(), ETHERNET_HEADER_LEN as uint);

        spawn( proc() {
            let mut buf: ~[u8] = ~[0, ..128];
            match RawSocket::new(DataLinkProtocol(EthernetProtocol(interface))) {
                Ok(mut sock) => match sock.recvfrom(buf) {
                    Ok((len, Some(MacAddress(addr)))) => {
                        assert_eq!(buf.slice(0, packet.len()), packet.as_slice());
                        assert_eq!(len, packet.len());
                        assert_eq!(addr, interface.mac_address());
                    },
                    _ => fail!()
                },
                Err(_) => fail!()
            }
        });

        match RawSocket::new(DataLinkProtocol(EthernetProtocol(interface))) {
            Ok(mut sock) => match sock.sendto(packet, None) {
                Ok(res) => assert_eq!(res as uint, packet.len()),
                Err(_) => fail!()
            },
            Err(_) => fail!()
        }

    } #[cfg(hasroot)])

}
