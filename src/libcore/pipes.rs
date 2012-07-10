// Runtime support for pipes.

import unsafe::{forget, reinterpret_cast, transmute};
import either::{either, left, right};
import option::unwrap;

enum state {
    empty,
    full,
    blocked,
    terminated
}

type packet_header = {
    mut state: state,
    mut blocked_task: option<*rust_task>,
};

type packet<T: send> = {
    header: packet_header,
    mut payload: option<T>
};

fn packet<T: send>() -> *packet<T> unsafe {
    let p: *packet<T> = unsafe::transmute(~{
        header: {
            mut state: empty,
            mut blocked_task: none::<task::task>,
        },
        mut payload: none::<T>
    });
    p
}

#[abi = "rust-intrinsic"]
extern mod rusti {
    fn atomic_xchng(&dst: int, src: int) -> int;
    fn atomic_xchng_acq(&dst: int, src: int) -> int;
    fn atomic_xchng_rel(&dst: int, src: int) -> int;
}

type rust_task = libc::c_void;

extern mod rustrt {
    #[rust_stack]
    fn rust_get_task() -> *rust_task;

    #[rust_stack]
    fn task_clear_event_reject(task: *rust_task);

    fn task_wait_event(this: *rust_task, killed: &mut bool) -> *libc::c_void;
    fn task_signal_event(target: *rust_task, event: *libc::c_void);
}

// We should consider moving this to core::unsafe, although I
// suspect graydon would want us to use void pointers instead.
unsafe fn uniquify<T>(x: *T) -> ~T {
    unsafe { unsafe::reinterpret_cast(x) }
}

fn wait_event(this: *rust_task) -> *libc::c_void {
    let mut killed = false;

    let res = rustrt::task_wait_event(this, &mut killed);
    if killed && !task::failing() {
        fail "killed"
    }
    res
}

fn swap_state_acq(&dst: state, src: state) -> state {
    unsafe {
        reinterpret_cast(rusti::atomic_xchng_acq(
            *(ptr::mut_addr_of(dst) as *mut int),
            src as int))
    }
}

fn swap_state_rel(&dst: state, src: state) -> state {
    unsafe {
        reinterpret_cast(rusti::atomic_xchng_rel(
            *(ptr::mut_addr_of(dst) as *mut int),
            src as int))
    }
}

fn send<T: send>(-p: send_packet<T>, -payload: T) {
    let p_ = p.unwrap();
    let p = unsafe { uniquify(p_) };
    assert (*p).payload == none;
    (*p).payload <- some(payload);
    let old_state = swap_state_rel(p.header.state, full);
    alt old_state {
      empty {
        // Yay, fastpath.

        // The receiver will eventually clean this up.
        unsafe { forget(p); }
      }
      full { fail "duplicate send" }
      blocked {
        #debug("waking up task for %?", p_);
        alt p.header.blocked_task {
          some(task) {
            rustrt::task_signal_event(
                task, ptr::addr_of(p.header) as *libc::c_void);
          }
          none { fail "blocked packet has no task" }
        }

        // The receiver will eventually clean this up.
        unsafe { forget(p); }
      }
      terminated {
        // The receiver will never receive this. Rely on drop_glue
        // to clean everything up.
      }
    }
}

fn recv<T: send>(-p: recv_packet<T>) -> T {
    option::unwrap(try_recv(p))
}

fn try_recv<T: send>(-p: recv_packet<T>) -> option<T> {
    let p_ = p.unwrap();
    let p = unsafe { uniquify(p_) };
    let this = rustrt::rust_get_task();
    rustrt::task_clear_event_reject(this);
    p.header.blocked_task = some(this);
    let mut first = true;
    loop {
        rustrt::task_clear_event_reject(this);
        let old_state = swap_state_acq(p.header.state,
                                       blocked);
        alt old_state {
          empty {
            #debug("no data available on %?, going to sleep.", p_);
            wait_event(this);
            #debug("woke up, p.state = %?", p.header.state);
          }
          blocked {
            if first {
                fail "blocking on already blocked packet"
            }
          }
          full {
            let mut payload = none;
            payload <-> (*p).payload;
            p.header.state = terminated;
            ret some(option::unwrap(payload))
          }
          terminated {
            assert old_state == terminated;
            ret none;
          }
        }
        first = false;
    }
}

/// Returns true if messages are available.
pure fn peek<T: send>(p: recv_packet<T>) -> bool {
    alt p.header().state {
      empty { false }
      blocked { fail "peeking on blocked packet" }
      full | terminated { true }
    }
}

fn sender_terminate<T: send>(p: *packet<T>) {
    let p = unsafe { uniquify(p) };
    alt swap_state_rel(p.header.state, terminated) {
      empty {
        // The receiver will eventually clean up.
        unsafe { forget(p) }
      }
      blocked {
        // wake up the target
        let target = p.header.blocked_task.get();
        rustrt::task_signal_event(target,
                                  ptr::addr_of(p.header) as *libc::c_void);

        // The receiver will eventually clean up.
        unsafe { forget(p) }
      }
      full {
        // This is impossible
        fail "you dun goofed"
      }
      terminated {
        // I have to clean up, use drop_glue
      }
    }
}

fn receiver_terminate<T: send>(p: *packet<T>) {
    let p = unsafe { uniquify(p) };
    alt swap_state_rel(p.header.state, terminated) {
      empty {
        // the sender will clean up
        unsafe { forget(p) }
      }
      blocked {
        // this shouldn't happen.
        fail "terminating a blocked packet"
      }
      terminated | full {
        // I have to clean up, use drop_glue
      }
    }
}

impl private_methods for packet_header {
    // Returns the old state.
    fn mark_blocked(this: *rust_task) -> state {
        self.blocked_task = some(this);
        swap_state_acq(self.state, blocked)
    }

    fn unblock() {
        alt swap_state_acq(self.state, empty) {
          empty | blocked { }
          terminated { self.state = terminated; }
          full { self.state = full; }
        }
    }
}

#[doc = "Returns when one of the packet headers reports data is
available."]
fn wait_many(pkts: &[&a.packet_header]) -> uint {
    let this = rustrt::rust_get_task();

    rustrt::task_clear_event_reject(this);
    let mut data_avail = false;
    let mut ready_packet = pkts.len();
    for pkts.eachi |i, p| {
        let old = p.mark_blocked(this);
        alt old {
          full | terminated {
            data_avail = true;
            ready_packet = i;
            p.state = old;
            break;
          }
          blocked { fail "blocking on blocked packet" }
          empty { }
        }
    }

    while !data_avail {
        #debug("sleeping on %? packets", pkts.len());
        let event = wait_event(this) as *packet_header;
        let pos = vec::position(pkts, |p| ptr::addr_of(*p) == event);

        alt pos {
          some(i) {
            ready_packet = i;
            data_avail = true;
          }
          none {
            #debug("ignoring spurious event, %?", event);
          }
        }
    }

    #debug("%?", pkts[ready_packet]);

    for pkts.each |p| { p.unblock() }

    #debug("%?, %?", ready_packet, pkts[ready_packet]);

    assert pkts[ready_packet].state == full
        || pkts[ready_packet].state == terminated;

    ready_packet
}

fn select2<A: send, B: send>(
    +a: recv_packet<A>,
    +b: recv_packet<B>)
    -> either<(option<A>, recv_packet<B>), (recv_packet<A>, option<B>)>
{
    let i = wait_many([a.header(), b.header()]/_);

    unsafe {
        alt i {
          0 { left((try_recv(a), b)) }
          1 { right((a, try_recv(b))) }
          _ { fail "select2 return an invalid packet" }
        }
    }
}

fn selecti<T: send>(endpoints: &[&recv_packet<T>]) -> uint {
    wait_many(endpoints.map(|p| p.header()))
}

#[doc = "Waits on a set of endpoints. Returns a message, its index,
 and a list of the remaining endpoints."]
fn select<T: send>(+endpoints: ~[recv_packet<T>])
    -> (uint, option<T>, ~[recv_packet<T>])
{
    let ready = wait_many(endpoints.map(|p| p.header()));
    let mut remaining = ~[];
    let mut result = none;
    do vec::consume(endpoints) |i, p| {
        if i == ready {
            result = try_recv(p);
        }
        else {
            vec::push(remaining, p);
        }
    }

    (ready, result, remaining)
}

class send_packet<T: send> {
    let mut p: option<*packet<T>>;
    new(p: *packet<T>) {
        //#debug("take send %?", p);
        self.p = some(p);
    }
    drop {
        //if self.p != none {
        //    #debug("drop send %?", option::get(self.p));
        //}
        if self.p != none {
            let mut p = none;
            p <-> self.p;
            sender_terminate(option::unwrap(p))
        }
    }
    fn unwrap() -> *packet<T> {
        let mut p = none;
        p <-> self.p;
        option::unwrap(p)
    }
}

class recv_packet<T: send> {
    let mut p: option<*packet<T>>;
    new(p: *packet<T>) {
        //#debug("take recv %?", p);
        self.p = some(p);
    }
    drop {
        //if self.p != none {
        //    #debug("drop recv %?", option::get(self.p));
        //}
        if self.p != none {
            let mut p = none;
            p <-> self.p;
            receiver_terminate(option::unwrap(p))
        }
    }
    fn unwrap() -> *packet<T> {
        let mut p = none;
        p <-> self.p;
        option::unwrap(p)
    }

    pure fn header() -> &self.packet_header {
        alt self.p {
          some(packet) {
            unsafe {
                let packet = uniquify(packet);
                let header = reinterpret_cast(&packet.header);
                forget(packet);
                header
            }
          }
          none { fail "packet already consumed" }
        }
    }
}

fn entangle<T: send>() -> (send_packet<T>, recv_packet<T>) {
    let p = packet();
    (send_packet(p), recv_packet(p))
}

fn spawn_service<T: send>(
    init: extern fn() -> (send_packet<T>, recv_packet<T>),
    +service: fn~(+recv_packet<T>))
    -> send_packet<T>
{
    let (client, server) = init();

    // This is some nasty gymnastics required to safely move the pipe
    // into a new task.
    let server = ~mut some(server);
    do task::spawn |move service| {
        let mut server_ = none;
        server_ <-> *server;
        service(option::unwrap(server_))
    }

    client
}

fn spawn_service_recv<T: send>(
    init: extern fn() -> (recv_packet<T>, send_packet<T>),
    +service: fn~(+send_packet<T>))
    -> recv_packet<T>
{
    let (client, server) = init();

    // This is some nasty gymnastics required to safely move the pipe
    // into a new task.
    let server = ~mut some(server);
    do task::spawn |move service| {
        let mut server_ = none;
        server_ <-> *server;
        service(option::unwrap(server_))
    }

    client
}

// Streams - Make pipes a little easier in general.

proto! streamp {
    open:send<T: send> {
        data(T) -> open<T>
    }
}

type chan<T:send> = { mut endp: option<streamp::client::open<T>> };
type port<T:send> = { mut endp: option<streamp::server::open<T>> };

fn stream<T:send>() -> (chan<T>, port<T>) {
    let (c, s) = streamp::init();

    #macro[
        [#move[x],
         unsafe { let y <- *ptr::addr_of(x); y }]
    ];

    ({ mut endp: some(c) }, { mut endp: some(s) })
}

impl chan<T: send> for chan<T> {
    fn send(+x: T) {
        let mut endp = none;
        endp <-> self.endp;
        self.endp = some(
            streamp::client::data(unwrap(endp), x))
    }
}

impl port<T: send> for port<T> {
    fn recv() -> T {
        let mut endp = none;
        endp <-> self.endp;
        let streamp::data(x, endp) = pipes::recv(unwrap(endp));
        self.endp = some(endp);
        x
    }

    fn try_recv() -> option<T> {
        let mut endp = none;
        endp <-> self.endp;
        alt pipes::try_recv(unwrap(endp)) {
          some(streamp::data(x, endp)) {
            self.endp = some(#move(endp));
            some(#move(x))
          }
          none { none }
        }
    }

    pure fn peek() -> bool unchecked {
        let mut endp = none;
        endp <-> self.endp;
        let peek = alt endp {
          some(endp) {
            pipes::peek(endp)
          }
          none { fail "peeking empty stream" }
        };
        self.endp <-> endp;
        peek
    }
}
