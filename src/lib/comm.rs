import sys;

export _chan;
export _port;

export mk_port;

native "rust" mod rustrt {
    type rust_chan;
    type rust_port;

    fn new_chan(po : *rust_port) -> *rust_chan;
    fn del_chan(ch : *rust_chan);
    fn drop_chan(ch : *rust_chan);

    fn new_port(unit_sz : uint) -> *rust_port;
    fn del_port(po : *rust_port);
    fn drop_port(po : *rust_port);
}

resource chan_ptr(ch: *rustrt::rust_chan) {
    rustrt::drop_chan(ch);
    rustrt::drop_chan(ch); // FIXME: We shouldn't have to do this
                           // twice.
    rustrt::del_chan(ch);
}

resource port_ptr(po: *rustrt::rust_port) {
    rustrt::drop_port(po);
    rustrt::del_port(po);
}

obj _chan[T](raw_chan : @chan_ptr) {
    fn send(v : &T) {

    }
}

obj _port[T](raw_port : @port_ptr) {
    fn mk_chan() -> _chan[T] {
        _chan(@chan_ptr(rustrt::new_chan(**raw_port)))
    }

    fn recv_into(v : &T) {

    }
}

fn mk_port[T]() -> _port[T] {
    _port(@port_ptr(rustrt::new_port(sys::size_of[T]())))
}
