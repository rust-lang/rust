mod dns;

mod tcpstream;
pub use tcpstream::*;

mod tcplistener;
pub use tcplistener::*;

mod udp;
pub use udp::*;

// this structure needs to be synchronized with what's in net/src/api.rs
#[repr(C)]
#[derive(Debug)]
enum NetError {
    // Ok = 0,
    Unaddressable = 1,
    SocketInUse = 2,
    // AccessDenied = 3,
    Invalid = 4,
    // Finished = 5,
    LibraryError = 6,
    // AlreadyUsed = 7,
    TimedOut = 8,
    WouldBlock = 9,
}

#[repr(C, align(4096))]
struct ConnectRequest {
    raw: [u8; 4096],
}

#[repr(C, align(4096))]
struct SendData {
    raw: [u8; 4096],
}

#[repr(C, align(4096))]
pub struct ReceiveData {
    raw: [u8; 4096],
}

#[repr(C, align(4096))]
pub struct GetAddress {
    raw: [u8; 4096],
}

pub use dns::LookupHost;

#[allow(nonstandard_style)]
pub mod netc {
    pub const AF_INET: u8 = 0;
    pub const AF_INET6: u8 = 1;
    pub type sa_family_t = u8;

    #[derive(Copy, Clone)]
    pub struct in_addr {
        pub s_addr: u32,
    }

    #[derive(Copy, Clone)]
    pub struct sockaddr_in {
        #[allow(dead_code)]
        pub sin_family: sa_family_t,
        pub sin_port: u16,
        pub sin_addr: in_addr,
    }

    #[derive(Copy, Clone)]
    pub struct in6_addr {
        pub s6_addr: [u8; 16],
    }

    #[derive(Copy, Clone)]
    pub struct sockaddr_in6 {
        #[allow(dead_code)]
        pub sin6_family: sa_family_t,
        pub sin6_port: u16,
        pub sin6_addr: in6_addr,
        pub sin6_flowinfo: u32,
        pub sin6_scope_id: u32,
    }
}
