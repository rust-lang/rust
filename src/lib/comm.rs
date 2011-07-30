import sys;

export _chan;
export _port;

export mk_port;
export mk_chan;

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

tag _chan[T] { _chan(@chan_dtor); }

resource port_ptr(po: *rustrt::rust_port) {
    rustrt::drop_port(po);
    rustrt::del_port(po);
}

tag _port[T] { _port(@port_dtor); }

fn mk_port[T]() -> _port[T] {
    _port(@port_dtor(rustrt::new_port(sys::size_of[T]())))
}

fn mk_chan[T](po : &_port[T]) -> _chan[T] {
    alt po {
      _port(_po) {
        _chan(@chan_dtor(rustrt::new_chan(**_po)))
      }
    }
}
