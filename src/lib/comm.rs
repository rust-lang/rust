import sys;
import ptr;
import unsafe;
import task;
import task::task_id;

export _chan;
export _port;

export mk_port;
export send;
export recv;
export chan;
export port;

native "rust" mod rustrt {
    type void;
    type rust_port;

    fn chan_id_send<~T>(target_task : task_id, target_port : port_id,
                        data : -T);

    fn new_port(unit_sz : uint) -> *rust_port;
    fn del_port(po : *rust_port);
    fn drop_port(po : *rust_port);
    fn get_port_id(po : *rust_port) -> port_id;
}

native "rust-intrinsic" mod rusti {
    fn recv<~T>(port : *rustrt::rust_port) -> T;
}

type port_id = int;

type chan<~T> = {
    task : task_id,
    port : port_id
};
type _chan<~T> = chan<T>;

resource port_ptr(po: *rustrt::rust_port) {
    rustrt::drop_port(po);
    rustrt::del_port(po);
}

type port<~T> = @port_ptr;

obj port_obj<~T>(raw_port : port<T>) {
    fn mk_chan() -> _chan<T> {
        chan::<T>(raw_port)
    }

    fn recv() -> T {
        recv(raw_port)
    }
}
type _port<~T> = port_obj<T>;

fn mk_port<~T>() -> _port<T> {
    ret port_obj::<T>(port::<T>());
}

fn send<~T>(ch : chan<T>, data : -T) {
    rustrt::chan_id_send(ch.task, ch.port, data);
}

fn port<~T>() -> port<T> {
    @port_ptr(rustrt::new_port(sys::size_of::<T>()))
}

fn recv<~T>(p : port<T>) -> T {
    ret rusti::recv(**p)
}

fn chan<~T>(p : port<T>) -> chan<T> {
    {
        task: task::get_task_id(),
        port: rustrt::get_port_id(**p)
    }
}
