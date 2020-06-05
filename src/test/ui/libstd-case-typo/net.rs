// checks case typos with libstd::net structs
fn main(){}

fn test_inc(_x: incoming){}
//~^ ERROR: cannot find type `incoming` in this scope
fn test_ipv4(_x: IPv4Addr){}
//~^ ERROR: cannot find type `IPv4Addr` in this scope
fn test_ipv6(_x: IPv6Addr){}
//~^ ERROR: cannot find type `IPv6Addr` in this scope
fn test_socv4(_x: SocketAddrv4){}
//~^ ERROR: cannot find type `SocketAddrv4` in this scope
fn test_socv6(_x: SocketAddrv6){}
//~^ ERROR: cannot find type `SocketAddrv6` in this scope
fn test_tcplist(_x: TCPListener){}
//~^ ERROR: cannot find type `TCPListener` in this scope
fn test_tcpstr(_x: TCPStream){}
//~^ ERROR: cannot find type `TCPStream` in this scope
fn test_udpsoc(_x: UDPSocket){}
//~^ ERROR: cannot find type `UDPSocket` in this scope
