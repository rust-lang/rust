import sys;
import ptr;
import unsafe;

export _chan;
export _port;

export mk_port;
export chan_from_unsafe_ptr;

native "rust" mod rustrt {
    type void;
    type rust_chan;
    type rust_port;

    fn new_chan(po : *rust_port) -> *rust_chan;
    fn take_chan(ch : *rust_chan);
    fn drop_chan(ch : *rust_chan);
    fn chan_send(ch: *rust_chan, v : *void);

    fn new_port(unit_sz : uint) -> *rust_port;
    fn del_port(po : *rust_port);
    fn drop_port(po : *rust_port);
}

native "rust-intrinsic" mod rusti {
    fn recv[T](port : *rustrt::rust_port) -> T;
}

resource chan_ptr(ch: *rustrt::rust_chan) {
    rustrt::drop_chan(ch);
}

resource port_ptr(po: *rustrt::rust_port) {
    rustrt::drop_port(po);
    rustrt::del_port(po);
}

obj _chan[T](raw_chan : @chan_ptr) {
    fn send(v : &T) {
        rustrt::chan_send(**raw_chan,
                          unsafe::reinterpret_cast(ptr::addr_of(v)));
    }

    // Use this to get something we can send over a channel.
    fn unsafe_ptr() -> *u8 {
        rustrt::take_chan(**raw_chan);
        ret unsafe::reinterpret_cast(**raw_chan);
    }
}

fn chan_from_unsafe_ptr[T](ch : *u8) -> _chan[T] {
    _chan(@chan_ptr(unsafe::reinterpret_cast(ch)))
}

obj _port[T](raw_port : @port_ptr) {
    fn mk_chan() -> _chan[T] {
        _chan(@chan_ptr(rustrt::new_chan(**raw_port)))
    }

    fn recv() -> T {
        ret rusti::recv(**raw_port)
    }
}

fn mk_port[T]() -> _port[T] {
    _port(@port_ptr(rustrt::new_port(sys::size_of[T]())))
}
