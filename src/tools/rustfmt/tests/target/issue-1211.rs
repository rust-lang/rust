fn main() {
    for iface in &ifaces {
        match iface.addr {
            get_if_addrs::IfAddr::V4(ref addr) => match addr.broadcast {
                Some(ip) => {
                    sock.send_to(&buf, (ip, 8765)).expect("foobar");
                }
                _ => (),
            },
            _ => (),
        };
    }
}
