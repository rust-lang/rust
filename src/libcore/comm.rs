// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];
/*!
 * Communication between tasks
 *
 * Communication between tasks is facilitated by ports (in the receiving
 * task), and channels (in the sending task). Any number of channels may
 * feed into a single port.  Ports and channels may only transmit values
 * of unique types; that is, values that are statically guaranteed to be
 * accessed by a single 'owner' at a time.  Unique types include scalars,
 * vectors, strings, and records, tags, tuples and unique boxes (`~T`)
 * thereof. Most notably, shared boxes (`@T`) may not be transmitted
 * across channels.
 *
 * # Example
 *
 * ~~~
 * let po = comm::port();
 * let ch = comm::chan(po);
 *
 * do task::spawn {
 *     comm::send(ch, "Hello, World");
 * }
 *
 * io::println(comm::recv(p));
 * ~~~
 */

import either::Either;
import libc::size_t;

export Port, port;
export Chan, chan;
export send;
export recv;
export peek;
export recv_chan;
export select2;
export methods;
export listen;


/**
 * A communication endpoint that can receive messages
 *
 * Each port has a unique per-task identity and may not be replicated or
 * transmitted. If a port value is copied, both copies refer to the same
 * port.  Ports may be associated with multiple `chan`s.
 */
enum Port<T: send> {
    Port_(@PortPtr<T>)
}

// It's critical that this only have one variant, so it has a record
// layout, and will work in the rust_task structure in task.rs.
/**
 * A communication endpoint that can send messages
 *
 * Each channel is bound to a port when the channel is constructed, so
 * the destination port for a channel must exist before the channel
 * itself.  Channels are weak: a channel does not keep the port it is
 * bound to alive. If a channel attempts to send data to a dead port that
 * data will be silently dropped.  Channels may be duplicated and
 * themselves transmitted over other channels.
 */
enum Chan<T: send> {
    Chan_(port_id)
}

/// Constructs a port
fn port<T: send>() -> Port<T> {
    Port_(@PortPtr(rustrt::new_port(sys::size_of::<T>() as size_t)))
}

impl<T: send> Port<T> {

    fn chan() -> Chan<T> { chan(self) }
    fn send(+v: T) { self.chan().send(v) }
    fn recv() -> T { recv(self) }
    fn peek() -> bool { peek(self) }

}

impl<T: send> Chan<T> {

    fn chan() -> Chan<T> { self }
    fn send(+v: T) { send(self, v) }
    fn recv() -> T { recv_chan(self) }
    fn peek() -> bool { peek_chan(self) }

}

/// Open a new receiving channel for the duration of a function
fn listen<T: send, U>(f: fn(Chan<T>) -> U) -> U {
    let po = port();
    f(po.chan())
}

struct PortPtr<T:send> {
  let po: *rust_port;
  new(po: *rust_port) { self.po = po; }
  drop unsafe {
      do task::unkillable {
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

/**
 * Internal function for converting from a channel to a port
 *
 * # Failure
 *
 * Fails if the port is detached or dead. Fails if the port
 * is owned by a different task.
 */
fn as_raw_port<T: send, U>(ch: comm::Chan<T>, f: fn(*rust_port) -> U) -> U {

    struct PortRef {
       let p: *rust_port;
       new(p: *rust_port) { self.p = p; }
       drop {
         if !ptr::is_null(self.p) {
           rustrt::rust_port_drop(self.p);
         }
       }
    }

    let p = PortRef(rustrt::rust_port_take(*ch));

    if ptr::is_null(p.p) {
        fail ~"unable to locate port for channel"
    } else if rustrt::get_task_id() != rustrt::rust_port_task(p.p) {
        fail ~"unable to access unowned port"
    }

    f(p.p)
}

/**
 * Constructs a channel. The channel is bound to the port used to
 * construct it.
 */
fn chan<T: send>(p: Port<T>) -> Chan<T> {
    Chan_(rustrt::get_port_id((**p).po))
}

/**
 * Sends data over a channel. The sent data is moved into the channel,
 * whereupon the caller loses access to it.
 */
fn send<T: send>(ch: Chan<T>, +data: T) {
    let Chan_(p) = ch;
    let data_ptr = ptr::addr_of(data) as *();
    let res = rustrt::rust_port_id_send(p, data_ptr);
    if res != 0u unsafe {
        // Data sent successfully
        unsafe::forget(data);
    }
    task::yield();
}

/**
 * Receive from a port.  If no data is available on the port then the
 * task will block until data becomes available.
 */
fn recv<T: send>(p: Port<T>) -> T { recv_((**p).po) }

/// Returns true if there are messages available
fn peek<T: send>(p: Port<T>) -> bool { peek_((**p).po) }

#[doc(hidden)]
fn recv_chan<T: send>(ch: comm::Chan<T>) -> T {
    as_raw_port(ch, |x|recv_(x))
}

fn peek_chan<T: send>(ch: comm::Chan<T>) -> bool {
    as_raw_port(ch, |x|peek_(x))
}

/// Receive on a raw port pointer
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
    return res;
}

fn peek_(p: *rust_port) -> bool {
    // Yield here before we check to see if someone sent us a message
    // FIXME #524, if the compilergenerates yields, we don't need this
    task::yield();
    rustrt::rust_port_size(p) != 0u as libc::size_t
}

/// Receive on one of two ports
fn select2<A: send, B: send>(p_a: Port<A>, p_b: Port<B>)
    -> Either<A, B> {
    let ports = ~[(**p_a).po, (**p_b).po];
    let yield = 0u, yieldp = ptr::addr_of(yield);

    let mut resport: *rust_port;
    resport = rusti::init::<*rust_port>();
    do vec::as_buf(ports) |ports, n_ports| {
        rustrt::rust_port_select(ptr::addr_of(resport), ports,
                                 n_ports as size_t, yieldp);
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
        either::Left(recv(p_a))
    } else if resport == (**p_b).po {
        either::Right(recv(p_b))
    } else {
        fail ~"unexpected result from rust_port_select";
    }
}


/* Implementation details */

#[allow(non_camel_case_types)] // runtime type
enum rust_port {}

#[allow(non_camel_case_types)] // runtime type
type port_id = int;

#[abi = "cdecl"]
extern mod rustrt {
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
extern mod rusti {
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
    let p = port::<Chan<int>>(), p2 = port::<int>();
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

    send(ch_a, ~"a");

    assert select2(po_a, po_b) == either::Left(~"a");

    send(ch_b, ~"b");

    assert select2(po_a, po_b) == either::Right(~"b");
}

#[test]
fn test_select2_rendezvous() {
    let po_a = port();
    let po_b = port();
    let ch_a = chan(po_a);
    let ch_b = chan(po_b);

    for iter::repeat(10u) {
        do task::spawn {
            for iter::repeat(10u) { task::yield() }
            send(ch_a, ~"a");
        };

        assert select2(po_a, po_b) == either::Left(~"a");

        do task::spawn {
            for iter::repeat(10u) { task::yield() }
            send(ch_b, ~"b");
        };

        assert select2(po_a, po_b) == either::Right(~"b");
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

    for iter::repeat(times) {
        do task::spawn {
            for iter::repeat(msgs) {
                send(ch_a, ~"a")
            }
        };
        do task::spawn {
            for iter::repeat(msgs) {
                send(ch_b, ~"b")
            }
        };
    }

    let mut as = 0;
    let mut bs = 0;
    for iter::repeat(msgs * times * 2u) {
        match select2(po_a, po_b) {
          either::Left(~"a") => as += 1,
          either::Right(~"b") => bs += 1,
          _ => fail ~"test_select_2_stress failed"
        }
    }

    assert as == 400;
    assert bs == 400;
}

#[test]
fn test_recv_chan() {
    let po = port();
    let ch = chan(po);
    send(ch, ~"flower");
    assert recv_chan(ch) == ~"flower";
}

#[test]
#[should_fail]
#[ignore(cfg(windows))]
fn test_recv_chan_dead() {
    let ch = chan(port());
    send(ch, ~"flower");
    recv_chan(ch);
}

#[test]
#[ignore(cfg(windows))]
fn test_recv_chan_wrong_task() {
    let po = port();
    let ch = chan(po);
    send(ch, ~"flower");
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
        do task::spawn {
            parent.send(~"oatmeal-salad");
        }
        assert parent.recv() == ~"oatmeal-salad";
    }
}

#[test]
#[ignore(cfg(windows))]
fn test_port_detach_fail() {
    for iter::repeat(100u) {
        do task::spawn_unlinked {
            let po = port();
            let ch = po.chan();

            do task::spawn {
                fail;
            }

            do task::spawn {
                ch.send(());
            }
        }
    }
}
