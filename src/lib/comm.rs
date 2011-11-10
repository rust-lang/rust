/*
Module: comm

Communication between tasks

Communication between tasks is facilitated by ports (in the receiving task),
and channels (in the sending task). Any number of channels may feed into a
single port.

Ports and channels may only transmit values of unique types; that is,
values that are statically guaranteed to be accessed by a single
'owner' at a time.  Unique types include scalars, vectors, strings,
and records, tags, tuples and unique boxes (~T) thereof. Most notably,
shared boxes (@T) may not be transmitted across channels.

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

native "cdecl" mod rustrt {
    type void;
    type rust_port;

    fn chan_id_send<uniq T>(t: *sys::type_desc,
                            target_task: task::task, target_port: port_id,
                            -data: T);

    fn new_port(unit_sz: uint) -> *rust_port;
    fn del_port(po: *rust_port);
    fn drop_port(po: *rust_port);
    fn get_port_id(po: *rust_port) -> port_id;
}

native "rust-intrinsic" mod rusti {
    fn recv<uniq T>(port: *rustrt::rust_port) -> T;
}

type port_id = int;

// It's critical that this only have one variant, so it has a record
// layout, and will work in the rust_task structure in task.rs.
/*
Type: chan

A communication endpoint that can send messages. Channels send
messages to ports.

Each channel is bound to a port when the channel is constructed, so
the destination port for a channel must exist before the channel
itself.

Channels are weak: a channel does not keep the port it is bound to alive.
If a channel attempts to send data to a dead port that data will be silently
dropped.

Channels may be duplicated and themselves transmitted over other channels.
*/
tag chan<uniq T> {
    chan_t(task::task, port_id);
}

resource port_ptr(po: *rustrt::rust_port) {
    rustrt::drop_port(po);
    rustrt::del_port(po);
}

/*
Type: port

A communication endpoint that can receive messages. Ports receive
messages from channels.

Each port has a unique per-task identity and may not be replicated or
transmitted. If a port value is copied, both copies refer to the same port.

Ports may be associated with multiple <chan>s.
*/
tag port<uniq T> { port_t(@port_ptr); }

/*
Function: send

Sends data over a channel.

The sent data is moved into the channel, whereupon the caller loses access
to it.
*/
fn send<uniq T>(ch: chan<T>, -data: T) {
    let chan_t(t, p) = ch;
    rustrt::chan_id_send(sys::get_type_desc::<T>(), t, p, data);
    task::yield();
}

/*
Function: port

Constructs a port.
*/
fn port<uniq T>() -> port<T> {
    port_t(@port_ptr(rustrt::new_port(sys::size_of::<T>())))
}

/*
Function: recv

Receive from a port.

If no data is available on the port then the task will block until data
becomes available.
*/
fn recv<uniq T>(p: port<T>) -> T { ret rusti::recv(***p) }

/*
Function: chan

Constructs a channel.

The channel is bound to the port used to construct it.
*/
fn chan<uniq T>(p: port<T>) -> chan<T> {
    chan_t(task::get_task(), rustrt::get_port_id(***p))
}
