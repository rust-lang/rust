/*
Module: comm

Communication between tasks

Communication between tasks is facilitated by ports (in the receiving task),
and channels (in the sending task). Any number of channels may feed into a
single port.

Example:

> use std::task;
> use std::comm;
>
> let p = comm::port();
> task::spawn(comm::chan(p), fn (c: chan<str>) {
>   comm::send(c, "Hello, World");
> });
>
> log comm::recv(p);

*/

import sys;
import ptr;
import unsafe;
import task;

export send;
export recv;
export chan;
export port;

native "c-stack-cdecl" mod rustrt {
    type void;
    type rust_port;

    fn chan_id_send<unique T>(t: *sys::type_desc,
                              target_task: task::task, target_port: port_id,
                              -data: T);

    fn new_port(unit_sz: uint) -> *rust_port;
    fn del_port(po: *rust_port);
    fn drop_port(po: *rust_port);
    fn get_port_id(po: *rust_port) -> port_id;
}

native "rust-intrinsic" mod rusti {
    fn recv<unique T>(port: *rustrt::rust_port) -> T;
}

type port_id = int;

// It's critical that this only have one variant, so it has a record
// layout, and will work in the rust_task structure in task.rs.
/*
Type: chan

A handle through which data may be sent.

Each channel is associated with a single <port>.
*/
tag chan<unique T> {
    chan_t(task::task, port_id);
}

resource port_ptr(po: *rustrt::rust_port) {
    rustrt::drop_port(po);
    rustrt::del_port(po);
}

/*
Type: port

A handle through which data may be received.

Ports may be associated with multiple <chan>s.
*/
tag port<unique T> { port_t(@port_ptr); }

/*
Function: send

Sends data over a channel.

The sent data is moved into the channel, whereupon the caller loses access
to it.
*/
fn send<unique T>(ch: chan<T>, -data: T) {
    let chan_t(t, p) = ch;
    rustrt::chan_id_send(sys::get_type_desc::<T>(), t, p, data);
    task::yield();
}

/*
Function: port

Constructs a port.
*/
fn port<unique T>() -> port<T> {
    port_t(@port_ptr(rustrt::new_port(sys::size_of::<T>())))
}

/*
Function: recv

Receive from a port.
*/
fn recv<unique T>(p: port<T>) -> T { ret rusti::recv(***p) }

/*
Function: chan

Constructs a channel.
*/
fn chan<unique T>(p: port<T>) -> chan<T> {
    chan_t(task::get_task_id(), rustrt::get_port_id(***p))
}
