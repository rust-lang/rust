import sys;
import ptr;
import unsafe;
import task;
import task::task_id;

export _chan;
export _port;

export mk_port;
export chan_from_unsafe_ptr;
export send;

native "rust" mod rustrt {
    type void;
    type rust_chan;
    type rust_port;

    fn new_chan(po : *rust_port) -> *rust_chan;
    fn take_chan(ch : *rust_chan);
    fn drop_chan(ch : *rust_chan);
    fn chan_send(ch: *rust_chan, v : *void);
    fn chan_id_send[~T](target_task : task_id, target_port : port_id,
                        data : -T);

    fn new_port(unit_sz : uint) -> *rust_port;
    fn del_port(po : *rust_port);
    fn drop_port(po : *rust_port);
    fn get_port_id(po : *rust_port) -> port_id;
}

native "rust-intrinsic" mod rusti {
    fn recv[~T](port : *rustrt::rust_port) -> T;
}

type port_id = int;

type _chan[~T] = {
    task : task_id,
    port : port_id
};

resource port_ptr(po: *rustrt::rust_port) {
    rustrt::drop_port(po);
    rustrt::del_port(po);
}

obj _port[~T](raw_port : @port_ptr) {
    // FIXME: rename this to chan once chan is not a keyword.
    fn mk_chan() -> _chan[T] {
        {
            task: task::get_task_id(),
            port: rustrt::get_port_id(**raw_port)
        }
    }

    fn recv() -> T {
        ret rusti::recv(**raw_port)
    }
}

fn mk_port[~T]() -> _port[T] {
    _port(@port_ptr(rustrt::new_port(sys::size_of[T]())))
}

fn send[~T](ch : _chan[T], data : -T) {
    rustrt::chan_id_send(ch.task, ch.port, data);
}
