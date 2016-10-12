pub use os::raw::*;

pub type size_t = usize;
pub type ssize_t = isize;

// Process/File definitions

pub type mode_t = u32;
pub type pid_t = u32;
pub type gid_t = u32;
pub type uid_t = u32;

pub const S_IFBLK: mode_t = 0;
pub const S_IFCHR: mode_t = 0;
pub const S_IFIFO: mode_t = 0;
pub const S_IFSOCK: mode_t = 0;

// Threading

#[derive(Copy,Clone)]
pub struct pthread_t(());
pub struct pthread_attr_t(());

// Time definitions

pub type time_t = i64;

#[derive(Copy,Clone)]
pub struct timespec {
    pub tv_sec: time_t,
    pub tv_nsec: c_long,
}

pub const CLOCK_MONOTONIC: c_int = 0;
pub const CLOCK_REALTIME: c_int = 0;

// Networking definitions

pub type sa_family_t = u16;
pub type in_port_t = u16;

#[derive(Copy,Clone)]
pub struct in_addr {
    pub s_addr: u32,
}

#[derive(Copy,Clone)]
pub struct sockaddr_in {
    pub sin_family: sa_family_t,
    pub sin_port: in_port_t,
    pub sin_addr: in_addr,
    pub sin_zero: [u8; 8],
}

#[derive(Copy,Clone)]
pub struct ip_mreq {
    pub imr_multiaddr: in_addr,
    pub imr_interface: in_addr,
}

#[derive(Copy,Clone)]
pub struct in6_addr {
    pub s6_addr: [u8; 16],
}

#[derive(Copy,Clone)]
pub struct sockaddr_in6 {
    pub sin6_family: sa_family_t,
    pub sin6_port: in_port_t,
    pub sin6_flowinfo: u32,
    pub sin6_addr: in6_addr,
    pub sin6_scope_id: u32,
}

#[derive(Copy,Clone)]
pub struct ipv6_mreq {
    pub ipv6mr_multiaddr: in6_addr,
    pub ipv6mr_interface: c_uint,
}

pub const AF_INET6: c_int = 0;
pub const AF_INET: c_int = 0;
pub const AF_UNIX: c_int = 0;
pub const IPPROTO_IPV6: c_int = 0;
pub const IPPROTO_IP: c_int = 0;
pub const IPV6_ADD_MEMBERSHIP: c_int = 0;
pub const IPV6_DROP_MEMBERSHIP: c_int = 0;
pub const IPV6_MULTICAST_LOOP: c_int = 0;
pub const IPV6_V6ONLY: c_int = 0;
pub const IP_ADD_MEMBERSHIP: c_int = 0;
pub const IP_DROP_MEMBERSHIP: c_int = 0;
pub const IP_MULTICAST_LOOP: c_int = 0;
pub const IP_MULTICAST_TTL: c_int = 0;
pub const IP_TTL: c_int = 0;
pub const SOCK_DGRAM: c_int = 0;
pub const SOCK_STREAM: c_int = 0;
pub const SOL_SOCKET: c_int = 0;
pub const SO_BROADCAST: c_int = 0;
pub const SO_RCVTIMEO: c_int = 0;
pub const SO_REUSEADDR: c_int = 0;
pub const SO_SNDTIMEO: c_int = 0;

pub struct sockaddr(());
#[derive(Clone)]
pub struct sockaddr_un(());
pub type socklen_t = u32;
pub struct sockaddr_storage(());

// C functions

pub unsafe fn strlen(s: *const c_char) -> size_t {
	let mut i=0isize;
	loop {
		if *s.offset(i)==0 {
			return i as usize
		}
		i+=1;
	}
}