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

import either::either;
import libc::size_t;

export port;
export chan;
export send;
export recv;
export peek;
export recv_chan;
export select2;
export methods;
export listen;


#[doc = "
A communication endpoint that can receive messages

Each port has a unique per-task identity and may not be replicated or
transmitted. If a port value is copied, both copies refer to the same
port.  Ports may be associated with multiple `chan`s.
"]
enum port<T: send> {
    port_t(@port_ptr<T>)
}

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
    chan_t(port_id)
}

#[doc = "Constructs a port"]
fn port<T: send>() -> port<T> {
    port_t(@port_ptr(rustrt::new_port(sys::size_of::<T>() as size_t)))
}

impl methods<T: send> for port<T> {

    fn chan() -> chan<T> { chan(self) }
    fn send(+v: T) { self.chan().send(v) }
    fn recv() -> T { recv(self) }
    fn peek() -> bool { peek(self) }

}

impl methods<T: send> for chan<T> {

    fn chan() -> chan<T> { self }
    fn send(+v: T) { send(self, v) }
    fn recv() -> T { recv_chan(self) }
    fn peek() -> bool { peek_chan(self) }

}

#[doc = "Open a new receiving channel for the duration of a function"]
fn listen<T: send, U>(f: fn(chan<T>) -> U) -> U {
    let po = port();
    f(po.chan())
}

class port_ptr<T:send> {
  let po: *rust_port;
  new(po: *rust_port) { self.po = po; }
  drop unsafe {
      do task::unkillable || {
        // Once the port is detached it's guaranteed not to receive further
        // messages
        let yield = 0u;
        let yieldp = ptr::addr_of(yield);
        rustrt::rust_port_begin_detach(self.po, yieldp);
        if yield != 0u {
            // Need to wait for the port to be detached
            task::yield();
        }
        rustrt::rust_port_end_detach(self.po);

        // Drain the port so that all the still-enqueued items get dropped
        while rustrt::rust_port_size(self.po) > 0u as size_t {
            recv_::<T>(self.po);
        }
        rustrt::del_port(self.po);
    }
  }
}

#[doc = "
Internal function for converting from a channel to a port

# Failure

Fails if the port is detached or dead. Fails if the port
is owned by a different task.
"]
fn as_raw_port<T: send, U>(ch: comm::chan<T>, f: fn(*rust_port) -> U) -> U {

    class portref {
       let p: *rust_port;
       new(p: *rust_port) { self.p = p; }
       drop {
         if !ptr::is_null(self.p) {
           rustrt::rust_port_drop(self.p);
         }
       }
    }

    let p = portref(rustrt::rust_port_take(*ch));

    if ptr::is_null(p.p) {
        fail "unable to locate port for channel"
    } else if rustrt::get_task_id() != rustrt::rust_port_task(p.p) {
        fail "unable to access unowned port"
    }

    f(p.p)
}

#[doc = "
Constructs a channel. The channel is bound to the port used to
construct it.
"]
fn chan<T: send>(p: port<T>) -> chan<T> {
    chan_t(rustrt::get_port_id((**p).po))
}

#[doc = "
Sends data over a channel. The sent data is moved into the channel,
whereupon the caller loses access to it.
"]
fn send<T: send>(ch: chan<T>, -data: T) {
    let chan_t(p) = ch;
    let data_ptr = ptr::addr_of(data) as *();
    let res = rustrt::rust_port_id_send(p, data_ptr);
    if res != 0u unsafe {
        // Data sent successfully
        unsafe::forget(data);
    }
    task::yield();
}

#[doc = "
Receive from a port.  If no data is available on the port then the
task will block until data becomes available.
"]
fn recv<T: send>(p: port<T>) -> T { recv_((**p).po) }

#[doc = "Returns true if there are messages available"]
fn peek<T: send>(p: port<T>) -> bool { peek_((**p).po) }

#[doc(hidden)]
fn recv_chan<T: send>(ch: comm::chan<T>) -> T {
    as_raw_port(ch, |x|recv_(x))
}

fn peek_chan<T: send>(ch: comm::chan<T>) -> bool {
    as_raw_port(ch, |x|peek_(x))
}

#[doc = "Receive on a raw port pointer"]
fn recv_<T: send>(p: *rust_port) -> T {
    let yield = 0u;
    let yieldp = ptr::addr_of(yield);
    let mut res;
    res = rusti::init::<T>();
    rustrt::port_recv(ptr::addr_of(res) as *uint, p, yieldp);

    if yield != 0u {
        // Data isn't available yet, so res has not been initialized.
        task::yield();
    } else {
        // In the absence of compiler-generated preemption points
        // this is a good place to yield
        task::yield();
    }
    ret res;
}

fn peek_(p: *rust_port) -> bool {
    rustrt::rust_port_size(p) != 0u as libc::size_t
}

#[doc = "Receive on one of two ports"]
fn select2<A: send, B: send>(p_a: port<A>, p_b: port<B>)
    -> either<A, B> {
    let ports = ~[(**p_a).po, (**p_b).po];
    let n_ports = 2 as libc::size_t;
    let yield = 0u, yieldp = ptr::addr_of(yield);

    let mut resport: *rust_port;
    resport = rusti::init::<*rust_port>();
    do vec::as_buf(ports) |ports| {
        rustrt::rust_port_select(ptr::addr_of(resport), ports, n_ports,
                                 yieldp);
    }

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

    if resport == (**p_a).po {
        either::left(recv(p_a))
    } else if resport == (**p_b).po {
        either::right(recv(p_b))
    } else {
        fail "unexpected result from rust_port_select";
    }
}


/* Implementation details */


enum rust_port {}

type port_id = int;

#[abi = "cdecl"]
native mod rustrt {
    fn rust_port_id_send(target_port: port_id, data: *()) -> libc::uintptr_t;

    fn new_port(unit_sz: libc::size_t) -> *rust_port;
    fn del_port(po: *rust_port);
    fn rust_port_begin_detach(po: *rust_port,
                              yield: *libc::uintptr_t);
    fn rust_port_end_detach(po: *rust_port);
    fn get_port_id(po: *rust_port) -> port_id;
    fn rust_port_size(po: *rust_port) -> libc::size_t;
    fn port_recv(dptr: *uint, po: *rust_port,
                 yield: *libc::uintptr_t);
    fn rust_port_select(dptr: **rust_port, ports: **rust_port,
                        n_ports: libc::size_t,
                        yield: *libc::uintptr_t);
    fn rust_port_take(port_id: port_id) -> *rust_port;
    fn rust_port_drop(p: *rust_port);
    fn rust_port_task(p: *rust_port) -> libc::uintptr_t;
    fn get_task_id() -> libc::uintptr_t;
}

#[abi = "rust-intrinsic"]
native mod rusti {
    fn init<T>() -> T;
}


/* Tests */


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

    do iter::repeat(10u) || {
        do task::spawn || {
            iter::repeat(10u, || task::yield());
            send(ch_a, "a");
        };

        assert select2(po_a, po_b) == either::left("a");

        do task::spawn || {
            iter::repeat(10u, || task::yield());
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

    do iter::repeat(times) || {
        do task::spawn || {
            do iter::repeat(msgs) || {
                send(ch_a, "a")
            }
        };
        do task::spawn || {
            do iter::repeat(msgs) || {
                send(ch_b, "b")
            }
        };
    }

    let mut as = 0;
    let mut bs = 0;
    do iter::repeat(msgs * times * 2u) || {
        alt check select2(po_a, po_b) {
          either::left("a") { as += 1 }
          either::right("b") { bs += 1 }
        }
    }

    assert as == 400;
    assert bs == 400;
}

#[test]
fn test_recv_chan() {
    let po = port();
    let ch = chan(po);
    send(ch, "flower");
    assert recv_chan(ch) == "flower";
}

#[test]
#[should_fail]
#[ignore(cfg(windows))]
fn test_recv_chan_dead() {
    let ch = chan(port());
    send(ch, "flower");
    recv_chan(ch);
}

#[test]
#[ignore(cfg(windows))]
fn test_recv_chan_wrong_task() {
    let po = port();
    let ch = chan(po);
    send(ch, "flower");
    assert result::is_err(task::try(||
        recv_chan(ch)
    ))
}

#[test]
fn test_port_send() {
    let po = port();
    po.send(());
    po.recv();
}

#[test]
fn test_chan_peek() {
    let po = port();
    let ch = po.chan();
    ch.send(());
    assert ch.peek();
}

#[test]
fn test_listen() {
    do listen |parent| {
        do task::spawn || {
            parent.send("oatmeal-salad");
        }
        assert parent.recv() == "oatmeal-salad";
    }
}

#[test]
#[ignore(cfg(windows))]
fn test_port_detach_fail() {
    do iter::repeat(100u) || {
        let builder = task::builder();
        task::unsupervise(builder);
        do task::run(builder) || {
            let po = port();
            let ch = po.chan();

            do task::spawn || {
                fail;
            }

            do task::spawn || {
                ch.send(());
            }
        }
    }
}
