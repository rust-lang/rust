extern {
    pub fn socket(domain: ::c_int, ty: ::c_int, protocol: ::c_int) -> ::c_int;
    pub fn bind(fd: ::c_int, addr: *const sockaddr, len: socklen_t) -> ::c_int;
    pub fn connect(socket: ::c_int, address: *const sockaddr,
                   len: socklen_t) -> ::c_int;
    pub fn listen(socket: ::c_int, backlog: ::c_int) -> ::c_int;
    pub fn getsockname(socket: ::c_int, address: *mut sockaddr,
                       address_len: *mut socklen_t) -> ::c_int;
    pub fn getsockopt(sockfd: ::c_int,
                      level: ::c_int,
                      optname: ::c_int,
                      optval: *mut ::c_void,
                      optlen: *mut ::socklen_t) -> ::c_int;
    pub fn setsockopt(socket: ::c_int, level: ::c_int, name: ::c_int,
                      value: *const ::c_void,
                      option_len: socklen_t) -> ::c_int;
    pub fn getpeername(socket: ::c_int, address: *mut sockaddr,
                       address_len: *mut socklen_t) -> ::c_int;
    pub fn sendto(socket: ::c_int, buf: *const ::c_void, len: ::size_t,
                  flags: ::c_int, addr: *const sockaddr,
                  addrlen: socklen_t) -> ::ssize_t;
    pub fn send(socket: ::c_int, buf: *const ::c_void, len: ::size_t,
                flags: ::c_int) -> ::ssize_t;
    pub fn recvfrom(socket: ::c_int, buf: *mut ::c_void, len: ::size_t,
                    flags: ::c_int, addr: *mut ::sockaddr,
                    addrlen: *mut ::socklen_t) -> ::ssize_t;
    pub fn recv(socket: ::c_int, buf: *mut ::c_void, len: ::size_t,
                flags: ::c_int) -> ::ssize_t;
}
