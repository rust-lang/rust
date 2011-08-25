import sys;
import ptr;
import unsafe;
import task;

export send;
export recv;
export chan;
export port;

native "rust" mod rustrt {
    type void;
    type rust_port;

    fn chan_id_send<~T>(target_task: task::task,
                        target_port: port_id, data: -T);

    fn new_port(unit_sz: uint) -> *rust_port;
    fn del_port(po: *rust_port);
    fn drop_port(po: *rust_port);
    fn get_port_id(po: *rust_port) -> port_id;
}

native "rust-intrinsic" mod rusti {
    fn recv<~T>(port: *rustrt::rust_port) -> T;
}

type port_id = int;

// It's critical that this only have one variant, so it has a record
// layout, and will work in the rust_task structure in task.rs.
tag chan<~T> { chan_t(task::task, port_id); }

resource port_ptr(po: *rustrt::rust_port) {
    rustrt::drop_port(po);
    rustrt::del_port(po);
}

tag port<~T> { port_t(@port_ptr); }

fn send<~T>(ch: &chan<T>, data: -T) {
    let chan_t(t, p) = ch;
    rustrt::chan_id_send(t, p, data);
}

fn port<~T>() -> port<T> {
    port_t(@port_ptr(rustrt::new_port(sys::size_of::<T>())))
}

fn recv<~T>(p: &port<T>) -> T { ret rusti::recv(***p) }

fn chan<~T>(p: &port<T>) -> chan<T> {
    chan_t(task::get_task_id(), rustrt::get_port_id(***p))
}
