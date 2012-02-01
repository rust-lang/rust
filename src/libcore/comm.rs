#[doc(
  brief = "Communication between tasks",
  desc  = "

Communication between tasks is facilitated by ports (in the receiving
task), and channels (in the sending task). Any number of channels may
feed into a single port.  Ports and channels may only transmit values
of unique types; that is, values that are statically guaranteed to be
accessed by a single 'owner' at a time.  Unique types include scalars,
vectors, strings, and records, tags, tuples and unique boxes (`~T`)
thereof. Most notably, shared boxes (`@T`) may not be transmitted
across channels.

Example:

    let p = comm::port();
    task::spawn(comm::chan(p), fn (c: chan<str>) {
        comm::send(c, \"Hello, World\");
    });
    io::println(comm::recv(p));

")];

import sys;
import task;

export send;
export recv;
export chan::{};
export port::{};

enum rust_port {}

#[abi = "cdecl"]
native mod rustrt {
    fn chan_id_send<T: send>(t: *sys::type_desc,
                            target_task: task::task, target_port: port_id,
                            data: T) -> ctypes::uintptr_t;

    fn new_port(unit_sz: ctypes::size_t) -> *rust_port;
    fn del_port(po: *rust_port);
    fn rust_port_detach(po: *rust_port);
    fn get_port_id(po: *rust_port) -> port_id;
    fn rust_port_size(po: *rust_port) -> ctypes::size_t;
    fn port_recv(dptr: *uint, po: *rust_port,
                 yield: *ctypes::uintptr_t,
                 killed: *ctypes::uintptr_t);
}

#[abi = "rust-intrinsic"]
native mod rusti {
    fn call_with_retptr<T: send>(&&f: fn@(*uint)) -> T;
}

type port_id = int;

// It's critical that this only have one variant, so it has a record
// layout, and will work in the rust_task structure in task.rs.
#[doc(
  brief = "A communication endpoint that can send messages. \
           Channels send messages to ports.",
  desc = "Each channel is bound to a port when the channel is \
          constructed, so the destination port for a channel \
          must exist before the channel itself. \
          Channels are weak: a channel does not keep the port it \
          is bound to alive. If a channel attempts to send data \
          to a dead port that data will be silently dropped. \
          Channels may be duplicated and themselves transmitted \
          over other channels."
)]
enum chan<T: send> {
    chan_t(task::task, port_id)
}

resource port_ptr<T: send>(po: *rust_port) {
    // Once the port is detached it's guaranteed not to receive further
    // messages
    rustrt::rust_port_detach(po);
    // Drain the port so that all the still-enqueued items get dropped
    while rustrt::rust_port_size(po) > 0u {
        // FIXME: For some reason if we don't assign to something here
        // we end up with invalid reads in the drop glue.
        let _t = recv_::<T>(po);
    }
    rustrt::del_port(po);
}

#[doc(
  brief = "A communication endpoint that can receive messages. \
           Ports receive messages from channels.",
  desc = "Each port has a unique per-task identity and may not \
          be replicated or transmitted. If a port value is \
          copied, both copies refer to the same port. \
          Ports may be associated with multiple <chan>s."
)]
enum port<T: send> { port_t(@port_ptr<T>) }

#[doc(
  brief = "Sends data over a channel. The sent data is moved \
           into the channel, whereupon the caller loses \
           access to it."
)]
fn send<T: send>(ch: chan<T>, -data: T) {
    let chan_t(t, p) = ch;
    let res = rustrt::chan_id_send(sys::get_type_desc::<T>(), t, p, data);
    if res != 0u unsafe {
        // Data sent successfully
        unsafe::leak(data);
    }
    task::yield();
}

#[doc(
  brief = "Constructs a port."
)]
fn port<T: send>() -> port<T> {
    port_t(@port_ptr(rustrt::new_port(sys::size_of::<T>())))
}

#[doc(
  brief = "Receive from a port. \
           If no data is available on the port then the task will \
           block until data becomes available."
)]
fn recv<T: send>(p: port<T>) -> T { recv_(***p) }

#[doc(
  brief = "Receive on a raw port pointer"
)]
fn recv_<T: send>(p: *rust_port) -> T {
    // FIXME: Due to issue 1185 we can't use a return pointer when
    // calling C code, and since we can't create our own return
    // pointer on the stack, we're going to call a little intrinsic
    // that will grab the value of the return pointer, then call this
    // function, which we will then use to call the runtime.
    fn recv(dptr: *uint, port: *rust_port,
            yield: *ctypes::uintptr_t,
            killed: *ctypes::uintptr_t) unsafe {
        rustrt::port_recv(dptr, port, yield, killed);
    }
    let yield = 0u;
    let yieldp = ptr::addr_of(yield);
    let killed = 0u;
    let killedp = ptr::addr_of(killed);
    let res = rusti::call_with_retptr(bind recv(_, p, yieldp, killedp));
    if killed != 0u {
        fail "killed";
    }
    if yield != 0u {
        // Data isn't available yet, so res has not been initialized.
        task::yield();
    }
    ret res;
}

#[doc(
  brief = "Constructs a channel. The channel is bound to the \
           port used to construct it."
)]
fn chan<T: send>(p: port<T>) -> chan<T> {
    chan_t(task::get_task(), rustrt::get_port_id(***p))
}

#[test]
fn create_port_and_chan() { let p = port::<int>(); chan(p); }

#[test]
fn send_int() {
    let p = port::<int>();
    let c = chan(p);
    send(c, 22);
}

#[test]
fn send_recv_fn() {
    let p = port::<int>();
    let c = chan::<int>(p);
    send(c, 42);
    assert (recv(p) == 42);
}

#[test]
fn send_recv_fn_infer() {
    let p = port();
    let c = chan(p);
    send(c, 42);
    assert (recv(p) == 42);
}

#[test]
fn chan_chan_infer() {
    let p = port(), p2 = port::<int>();
    let c = chan(p);
    send(c, chan(p2));
    recv(p);
}

#[test]
fn chan_chan() {
    let p = port::<chan<int>>(), p2 = port::<int>();
    let c = chan(p);
    send(c, chan(p2));
    recv(p);
}

