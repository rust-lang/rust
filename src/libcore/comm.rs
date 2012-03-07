#[doc = "
Communication between tasks

Communication between tasks is facilitated by ports (in the receiving
task), and channels (in the sending task). Any number of channels may
feed into a single port.  Ports and channels may only transmit values
of unique types; that is, values that are statically guaranteed to be
accessed by a single 'owner' at a time.  Unique types include scalars,
vectors, strings, and records, tags, tuples and unique boxes (`~T`)
thereof. Most notably, shared boxes (`@T`) may not be transmitted
across channels.

# Example

~~~
let po = comm::port();
let ch = comm::chan(po);

task::spawn {||
    comm::send(ch, \"Hello, World\");
});

io::println(comm::recv(p));
~~~
"];

import sys;
import task;

export send;
export recv;
export peek;
export select2;
export chan::{};
export port::{};

enum rust_port {}

#[abi = "cdecl"]
native mod rustrt {
    fn get_task_id() -> task_id;
    fn chan_id_send<T: send>(t: *sys::type_desc,
                            target_task: task_id, target_port: port_id,
                            data: T) -> ctypes::uintptr_t;

    fn new_port(unit_sz: ctypes::size_t) -> *rust_port;
    fn del_port(po: *rust_port);
    fn rust_port_begin_detach(po: *rust_port,
                              yield: *ctypes::uintptr_t);
    fn rust_port_end_detach(po: *rust_port);
    fn get_port_id(po: *rust_port) -> port_id;
    fn rust_port_size(po: *rust_port) -> ctypes::size_t;
    fn port_recv(dptr: *uint, po: *rust_port,
                 yield: *ctypes::uintptr_t);
    fn rust_port_select(dptr: **rust_port, ports: **rust_port,
                        n_ports: ctypes::size_t,
                        yield: *ctypes::uintptr_t);
}

#[abi = "rust-intrinsic"]
native mod rusti {
    // FIXME: This should probably not take a boxed closure
    fn call_with_retptr<T: send>(&&f: fn@(*uint)) -> T;
}

type task_id = int;
type port_id = int;

// It's critical that this only have one variant, so it has a record
// layout, and will work in the rust_task structure in task.rs.
#[doc = "
A communication endpoint that can send messages

Each channel is bound to a port when the channel is constructed, so
the destination port for a channel must exist before the channel
itself.  Channels are weak: a channel does not keep the port it is
bound to alive. If a channel attempts to send data to a dead port that
data will be silently dropped.  Channels may be duplicated and
themselves transmitted over other channels.
"]
enum chan<T: send> {
    chan_t(task_id, port_id)
}

resource port_ptr<T: send>(po: *rust_port) {
    // Once the port is detached it's guaranteed not to receive further
    // messages
    let yield = 0u;
    let yieldp = ptr::addr_of(yield);
    rustrt::rust_port_begin_detach(po, yieldp);
    if yield != 0u {
        // Need to wait for the port to be detached
        // FIXME: If this fails then we're going to leave our port
        // in a bogus state.
        task::yield();
    }
    rustrt::rust_port_end_detach(po);

    // Drain the port so that all the still-enqueued items get dropped
    while rustrt::rust_port_size(po) > 0u {
        // FIXME: For some reason if we don't assign to something here
        // we end up with invalid reads in the drop glue.
        let _t = recv_::<T>(po);
    }
    rustrt::del_port(po);
}

#[doc = "
A communication endpoint that can receive messages

Each port has a unique per-task identity and may not be replicated or
transmitted. If a port value is copied, both copies refer to the same
port.  Ports may be associated with multiple `chan`s.
"]
enum port<T: send> { port_t(@port_ptr<T>) }

#[doc = "
Sends data over a channel. The sent data is moved into the channel,
whereupon the caller loses access to it.
"]
fn send<T: send>(ch: chan<T>, -data: T) {
    let chan_t(t, p) = ch;
    let res = rustrt::chan_id_send(sys::get_type_desc::<T>(), t, p, data);
    if res != 0u unsafe {
        // Data sent successfully
        unsafe::leak(data);
    }
    task::yield();
}

#[doc = "Constructs a port"]
fn port<T: send>() -> port<T> {
    port_t(@port_ptr(rustrt::new_port(sys::size_of::<T>())))
}

#[doc = "
Receive from a port.  If no data is available on the port then the
task will block until data becomes available.
"]
fn recv<T: send>(p: port<T>) -> T { recv_(***p) }

#[doc = "Receive on a raw port pointer"]
fn recv_<T: send>(p: *rust_port) -> T {
    // FIXME: Due to issue 1185 we can't use a return pointer when
    // calling C code, and since we can't create our own return
    // pointer on the stack, we're going to call a little intrinsic
    // that will grab the value of the return pointer, then call this
    // function, which we will then use to call the runtime.
    fn recv(dptr: *uint, port: *rust_port,
            yield: *ctypes::uintptr_t) unsafe {
        rustrt::port_recv(dptr, port, yield);
    }
    let yield = 0u;
    let yieldp = ptr::addr_of(yield);
    let res = rusti::call_with_retptr(bind recv(_, p, yieldp));
    if yield != 0u {
        // Data isn't available yet, so res has not been initialized.
        task::yield();
    } else {
        // In the absense of compiler-generated preemption points
        // this is a good place to yield
        task::yield();
    }
    ret res;
}

#[doc = "Receive on one of two ports"]
fn select2<A: send, B: send>(
    p_a: port<A>, p_b: port<B>
) -> either::t<A, B> unsafe {

    fn select(dptr: **rust_port, ports: **rust_port,
              n_ports: ctypes::size_t, yield: *ctypes::uintptr_t) {
        rustrt::rust_port_select(dptr, ports, n_ports, yield)
    }

    let mut ports = [];
    ports += [***p_a, ***p_b];
    let n_ports = 2 as ctypes::size_t;
    let yield = 0u;
    let yieldp = ptr::addr_of(yield);

    let resport: *rust_port = vec::as_buf(ports) {|ports|
        rusti::call_with_retptr {|retptr|
            select(unsafe::reinterpret_cast(retptr), ports, n_ports, yieldp)
        }
    };

    if yield != 0u {
        // Wait for data
        task::yield();
    } else {
        // As in recv, this is a good place to yield anyway until
        // the compiler generates yield calls
        task::yield();
    }

    // Now we know the port we're supposed to receive from
    assert resport != ptr::null();

    if resport == ***p_a {
        either::left(recv(p_a))
    } else if resport == ***p_b {
        either::right(recv(p_b))
    } else {
        fail "unexpected result from rust_port_select";
    }
}

#[doc = "Returns true if there are messages available"]
fn peek<T: send>(p: port<T>) -> bool {
    rustrt::rust_port_size(***p) != 0u as ctypes::size_t
}

#[doc = "
Constructs a channel. The channel is bound to the port used to
construct it.
"]
fn chan<T: send>(p: port<T>) -> chan<T> {
    chan_t(rustrt::get_task_id(), rustrt::get_port_id(***p))
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

#[test]
fn test_peek() {
    let po = port();
    let ch = chan(po);
    assert !peek(po);
    send(ch, ());
    assert peek(po);
    recv(po);
    assert !peek(po);
}

#[test]
fn test_select2_available() {
    let po_a = port();
    let po_b = port();
    let ch_a = chan(po_a);
    let ch_b = chan(po_b);

    send(ch_a, "a");

    assert select2(po_a, po_b) == either::left("a");

    send(ch_b, "b");

    assert select2(po_a, po_b) == either::right("b");
}

#[test]
fn test_select2_rendezvous() {
    let po_a = port();
    let po_b = port();
    let ch_a = chan(po_a);
    let ch_b = chan(po_b);

    iter::repeat(10u) {||
        task::spawn {||
            iter::repeat(10u) {|| task::yield() }
            send(ch_a, "a");
        };

        assert select2(po_a, po_b) == either::left("a");

        task::spawn {||
            iter::repeat(10u) {|| task::yield() }
            send(ch_b, "b");
        };

        assert select2(po_a, po_b) == either::right("b");
    }
}

#[test]
fn test_select2_stress() {
    let po_a = port();
    let po_b = port();
    let ch_a = chan(po_a);
    let ch_b = chan(po_b);

    let msgs = 100u;
    let times = 4u;

    iter::repeat(times) {||
        task::spawn {||
            iter::repeat(msgs) {||
                send(ch_a, "a")
            }
        };
        task::spawn {||
            iter::repeat(msgs) {||
                send(ch_b, "b")
            }
        };
    }

    let as = 0;
    let bs = 0;
    iter::repeat(msgs * times * 2u) {||
        alt check select2(po_a, po_b) {
          either::left("a") { as += 1 }
          either::right("b") { bs += 1 }
        }
    }

    assert as == 400;
    assert bs == 400;
}
