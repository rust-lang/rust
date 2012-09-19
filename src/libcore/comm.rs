/*!

Deprecated communication between tasks

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
let po = comm::Port();
let ch = comm::Chan(po);

do task::spawn {
    comm::send(ch, "Hello, World");
}

io::println(comm::recv(p));
~~~

# Note

Use of this module is deprecated in favor of `core::pipes`. In the
`core::comm` will likely be rewritten with pipes, at which point it
will once again be the preferred module for intertask communication.

*/

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use either::Either;
use libc::size_t;



/**
 * A communication endpoint that can receive messages
 *
 * Each port has a unique per-task identity and may not be replicated or
 * transmitted. If a port value is copied, both copies refer to the same
 * port.  Ports may be associated with multiple `chan`s.
 */
pub enum Port<T: Send> {
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
pub enum Chan<T: Send> {
    Chan_(port_id)
}

/// Constructs a port
pub fn Port<T: Send>() -> Port<T> {
    Port_(@PortPtr(rustrt::new_port(sys::size_of::<T>() as size_t)))
}

impl<T: Send> Port<T> {

    fn chan() -> Chan<T> { Chan(self) }
    fn send(+v: T) { self.chan().send(move v) }
    fn recv() -> T { recv(self) }
    fn peek() -> bool { peek(self) }

}

impl<T: Send> Chan<T> {

    fn chan() -> Chan<T> { self }
    fn send(+v: T) { send(self, move v) }
    fn recv() -> T { recv_chan(self) }
    fn peek() -> bool { peek_chan(self) }

}

/// Open a new receiving channel for the duration of a function
pub fn listen<T: Send, U>(f: fn(Chan<T>) -> U) -> U {
    let po = Port();
    f(po.chan())
}

struct PortPtr<T:Send> {
    po: *rust_port,
  drop unsafe {
      do task::unkillable {
        // Once the port is detached it's guaranteed not to receive further
        // messages
        let yield = 0;
        let yieldp = ptr::addr_of(yield);
        rustrt::rust_port_begin_detach(self.po, yieldp);
        if yield != 0 {
            // Need to wait for the port to be detached
            task::yield();
        }
        rustrt::rust_port_end_detach(self.po);

        // Drain the port so that all the still-enqueued items get dropped
        while rustrt::rust_port_size(self.po) > 0 as size_t {
            recv_::<T>(self.po);
        }
        rustrt::del_port(self.po);
    }
  }
}

fn PortPtr<T: Send>(po: *rust_port) -> PortPtr<T> {
    PortPtr {
        po: po
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
fn as_raw_port<T: Send, U>(ch: comm::Chan<T>, f: fn(*rust_port) -> U) -> U {

    struct PortRef {
        p: *rust_port,
       drop {
         if !ptr::is_null(self.p) {
           rustrt::rust_port_drop(self.p);
         }
       }
    }

    fn PortRef(p: *rust_port) -> PortRef {
        PortRef {
            p: p
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
pub fn Chan<T: Send>(p: Port<T>) -> Chan<T> {
    Chan_(rustrt::get_port_id((**p).po))
}

/**
 * Sends data over a channel. The sent data is moved into the channel,
 * whereupon the caller loses access to it.
 */
pub fn send<T: Send>(ch: Chan<T>, +data: T) {
    let Chan_(p) = ch;
    let data_ptr = ptr::addr_of(data) as *();
    let res = rustrt::rust_port_id_send(p, data_ptr);
    if res != 0 unsafe {
        // Data sent successfully
        cast::forget(move data);
    }
    task::yield();
}

/**
 * Receive from a port.  If no data is available on the port then the
 * task will block until data becomes available.
 */
pub fn recv<T: Send>(p: Port<T>) -> T { recv_((**p).po) }

/// Returns true if there are messages available
pub fn peek<T: Send>(p: Port<T>) -> bool { peek_((**p).po) }

#[doc(hidden)]
pub fn recv_chan<T: Send>(ch: comm::Chan<T>) -> T {
    as_raw_port(ch, |x|recv_(x))
}

fn peek_chan<T: Send>(ch: comm::Chan<T>) -> bool {
    as_raw_port(ch, |x|peek_(x))
}

/// Receive on a raw port pointer
fn recv_<T: Send>(p: *rust_port) -> T {
    let yield = 0;
    let yieldp = ptr::addr_of(yield);
    let mut res;
    res = rusti::init::<T>();
    rustrt::port_recv(ptr::addr_of(res) as *uint, p, yieldp);

    if yield != 0 {
        // Data isn't available yet, so res has not been initialized.
        task::yield();
    } else {
        // In the absence of compiler-generated preemption points
        // this is a good place to yield
        task::yield();
    }
    move res
}

fn peek_(p: *rust_port) -> bool {
    // Yield here before we check to see if someone sent us a message
    // FIXME #524, if the compiler generates yields, we don't need this
    task::yield();
    rustrt::rust_port_size(p) != 0 as libc::size_t
}

/// Receive on one of two ports
pub fn select2<A: Send, B: Send>(p_a: Port<A>, p_b: Port<B>)
    -> Either<A, B> {
    let ports = ~[(**p_a).po, (**p_b).po];
    let yield = 0, yieldp = ptr::addr_of(yield);

    let mut resport: *rust_port;
    resport = rusti::init::<*rust_port>();
    do vec::as_imm_buf(ports) |ports, n_ports| {
        rustrt::rust_port_select(ptr::addr_of(resport), ports,
                                 n_ports as size_t, yieldp);
    }

    if yield != 0 {
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
fn create_port_and_chan() { let p = Port::<int>(); Chan(p); }

#[test]
fn send_int() {
    let p = Port::<int>();
    let c = Chan(p);
    send(c, 22);
}

#[test]
fn send_recv_fn() {
    let p = Port::<int>();
    let c = Chan::<int>(p);
    send(c, 42);
    assert (recv(p) == 42);
}

#[test]
fn send_recv_fn_infer() {
    let p = Port();
    let c = Chan(p);
    send(c, 42);
    assert (recv(p) == 42);
}

#[test]
fn chan_chan_infer() {
    let p = Port(), p2 = Port::<int>();
    let c = Chan(p);
    send(c, Chan(p2));
    recv(p);
}

#[test]
fn chan_chan() {
    let p = Port::<Chan<int>>(), p2 = Port::<int>();
    let c = Chan(p);
    send(c, Chan(p2));
    recv(p);
}

#[test]
fn test_peek() {
    let po = Port();
    let ch = Chan(po);
    assert !peek(po);
    send(ch, ());
    assert peek(po);
    recv(po);
    assert !peek(po);
}

#[test]
fn test_select2_available() {
    let po_a = Port();
    let po_b = Port();
    let ch_a = Chan(po_a);
    let ch_b = Chan(po_b);

    send(ch_a, ~"a");

    assert select2(po_a, po_b) == either::Left(~"a");

    send(ch_b, ~"b");

    assert select2(po_a, po_b) == either::Right(~"b");
}

#[test]
fn test_select2_rendezvous() {
    let po_a = Port();
    let po_b = Port();
    let ch_a = Chan(po_a);
    let ch_b = Chan(po_b);

    for iter::repeat(10) {
        do task::spawn {
            for iter::repeat(10) { task::yield() }
            send(ch_a, ~"a");
        };

        assert select2(po_a, po_b) == either::Left(~"a");

        do task::spawn {
            for iter::repeat(10) { task::yield() }
            send(ch_b, ~"b");
        };

        assert select2(po_a, po_b) == either::Right(~"b");
    }
}

#[test]
fn test_select2_stress() {
    let po_a = Port();
    let po_b = Port();
    let ch_a = Chan(po_a);
    let ch_b = Chan(po_b);

    let msgs = 100;
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

    let mut as_ = 0;
    let mut bs = 0;
    for iter::repeat(msgs * times * 2u) {
        match select2(po_a, po_b) {
          either::Left(~"a") => as_ += 1,
          either::Right(~"b") => bs += 1,
          _ => fail ~"test_select_2_stress failed"
        }
    }

    assert as_ == 400;
    assert bs == 400;
}

#[test]
fn test_recv_chan() {
    let po = Port();
    let ch = Chan(po);
    send(ch, ~"flower");
    assert recv_chan(ch) == ~"flower";
}

#[test]
#[should_fail]
#[ignore(cfg(windows))]
fn test_recv_chan_dead() {
    let ch = Chan(Port());
    send(ch, ~"flower");
    recv_chan(ch);
}

#[test]
#[ignore(cfg(windows))]
fn test_recv_chan_wrong_task() {
    let po = Port();
    let ch = Chan(po);
    send(ch, ~"flower");
    assert result::is_err(task::try(||
        recv_chan(ch)
    ))
}

#[test]
fn test_port_send() {
    let po = Port();
    po.send(());
    po.recv();
}

#[test]
fn test_chan_peek() {
    let po = Port();
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
    for iter::repeat(100) {
        do task::spawn_unlinked {
            let po = Port();
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
