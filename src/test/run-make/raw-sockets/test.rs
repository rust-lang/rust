extern mod native;

use std::io::net::raw::{RawSocket};
use std::io::net::ip::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::libc::{AF_INET, AF_INET6};


// RFC 3692 test protocol number
static testProtocol: i32 = 253;

fn test(ip: IpAddr) {
    let message = "message";
    let prot = match ip {
        Ipv4Addr(..) => AF_INET,
        Ipv6Addr(..) => AF_INET6
    };
    spawn( proc() {
        let mut buf: ~[u8] = ~[0, .. 128];
        let sock = RawSocket::new(prot, testProtocol, false);
        match sock {
            Ok(mut s) => match s.recvfrom(buf) {
                            Ok((len, addr)) => {
                                assert_eq!(buf.slice(0, message.len()), message.as_bytes());
                                assert_eq!(len, message.len());
                                assert_eq!(addr, ip);
                            },
                            Err(_) => fail!()
                         },
            Err(_) => fail!()
        };
    });

    let sock = RawSocket::new(prot, testProtocol, false);
    let _res = sock.map(|mut sock| {
        match sock.sendto(message.as_bytes(), ip) {
            Ok(res) => assert_eq!(res as uint, message.len()),
            Err(_) => fail!()
        }
    });
}

fn ipv4_test() {
    test(Ipv4Addr(127, 0, 0, 1));
}

fn ipv6_test() {
    test(Ipv6Addr(0, 0, 0, 0, 0, 0, 0, 1));
}

#[test]
fn native_ipv4_test() {
    let (p, c) = Chan::new();
    native::task::spawn(proc() { c.send(ipv4_test()) });
    p.recv();
}

#[test]
fn native_ipv6_test() {
    let (p, c) = Chan::new();
    native::task::spawn(proc() { c.send(ipv6_test()) });
    p.recv();
}
