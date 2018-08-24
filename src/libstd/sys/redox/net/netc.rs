pub type in_addr_t = u32;
pub type in_port_t = u16;

pub type socklen_t = u32;
pub type sa_family_t = u16;

pub const AF_INET: sa_family_t = 2;
pub const AF_INET6: sa_family_t = 23;

#[derive(Copy, Clone)]
#[repr(C)]
pub struct in_addr {
    pub s_addr: in_addr_t,
}

#[derive(Copy, Clone)]
#[repr(align(4))]
#[repr(C)]
pub struct in6_addr {
    pub s6_addr: [u8; 16],
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct sockaddr {
    pub sa_family: sa_family_t,
    pub sa_data: [u8; 14],
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct sockaddr_in {
    pub sin_family: sa_family_t,
    pub sin_port: in_port_t,
    pub sin_addr: in_addr,
    pub sin_zero: [u8; 8],
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct sockaddr_in6 {
    pub sin6_family: sa_family_t,
    pub sin6_port: in_port_t,
    pub sin6_flowinfo: u32,
    pub sin6_addr: in6_addr,
    pub sin6_scope_id: u32,
}
