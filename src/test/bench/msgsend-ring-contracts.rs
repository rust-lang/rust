// This test creates a bunch of tasks that simultaneously send to each
// other in a ring. The messages should all be basically
// independent. It's designed to hammer the global kernel lock, so
// that things will look really good once we get that lock out of the
// message path.

// This version uses semi-automatically compiled channel contracts.

// xfail-pretty

import future::future;

use std;
import std::time;

import ring::server::recv;

mod pipes {
    // Runtime support for pipes.

    import unsafe::{forget, reinterpret_cast};

    enum state {
        empty,
        full,
        blocked,
        terminated
    }

    type packet<T: send> = {
        mut state: state,
        mut blocked_task: option<task::task>,
        mut payload: option<T>
    };

    fn packet<T: send>() -> *packet<T> unsafe {
        let p: *packet<T> = unsafe::transmute(~{
            mut state: empty,
            mut blocked_task: none::<task::task>,
            mut payload: none::<T>
        });
        p
    }

    #[abi = "rust-intrinsic"]
    native mod rusti {
        fn atomic_xchng(&dst: int, src: int) -> int;
        fn atomic_xchng_acq(&dst: int, src: int) -> int;
        fn atomic_xchng_rel(&dst: int, src: int) -> int;
    }

    // We should consider moving this to core::unsafe, although I
    // suspect graydon would want us to use void pointers instead.
    unsafe fn uniquify<T>(x: *T) -> ~T {
        unsafe { unsafe::reinterpret_cast(x) }
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
        let p = p.unwrap();
        let p = unsafe { uniquify(p) };
        assert (*p).payload == none;
        (*p).payload <- some(payload);
        let old_state = swap_state_rel((*p).state, full);
        alt old_state {
          empty {
            // Yay, fastpath.

            // The receiver will eventually clean this up.
            unsafe { forget(p); }
          }
          full { fail "duplicate send" }
          blocked {
            // FIXME: once the target will actually block, tell the
            // scheduler to wake it up.

            // The receiver will eventually clean this up.
            unsafe { forget(p); }
          }
          terminated {
            // The receiver will never receive this. Rely on drop_glue
            // to clean everything up.
          }
        }
    }

    fn recv<T: send>(-p: recv_packet<T>) -> option<T> {
        let p = p.unwrap();
        let p = unsafe { uniquify(p) };
        loop {
            let old_state = swap_state_acq((*p).state,
                                           blocked);
            alt old_state {
              empty | blocked { task::yield(); }
              full {
                let mut payload = none;
                payload <-> (*p).payload;
                ret some(option::unwrap(payload))
              }
              terminated {
                assert old_state == terminated;
                ret none;
              }
            }
        }
    }

    fn sender_terminate<T: send>(p: *packet<T>) {
        let p = unsafe { uniquify(p) };
        alt swap_state_rel((*p).state, terminated) {
          empty | blocked {
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
        alt swap_state_rel((*p).state, terminated) {
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

    class send_packet<T: send> {
        let mut p: option<*packet<T>>;
        new(p: *packet<T>) {
            //#error("take send %?", p);
            self.p = some(p);
        }
        drop {
            //if self.p != none {
            //    #error("drop send %?", option::get(self.p));
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
            //#error("take recv %?", p);
            self.p = some(p);
        }
        drop {
            //if self.p != none {
            //    #error("drop recv %?", option::get(self.p));
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
    }

    fn entangle<T: send>() -> (send_packet<T>, recv_packet<T>) {
        let p = packet();
        (send_packet(p), recv_packet(p))
    }
}

// This module was generated by the pipe compiler.
mod ring {
    fn init() -> (client::num, server::num) { pipes::entangle() }
    enum num { num(uint, server::num), }
    mod client {
        fn num(-pipe: num, x_0: uint) -> num {
            let (c, s) = pipes::entangle();
            let message = ring::num(x_0, s);
            pipes::send(pipe, message);
            c
        }
        type num = pipes::send_packet<ring::num>;
    }
    mod server {
        impl recv for num {
            fn recv() -> extern fn(-num) -> ring::num {
                fn recv(-pipe: num) -> ring::num {
                    option::unwrap(pipes::recv(pipe))
                }
                recv
            }
        }
        type num = pipes::recv_packet<ring::num>;
    }
}

fn macros() {
    #macro[
        [#recv[chan],
         chan.recv()(chan)]
    ];

    #macro[
        [#move[x],
         unsafe { let y <- *ptr::addr_of(x); y }]
    ];
}

fn thread_ring(i: uint,
               count: uint,
               +num_chan: ring::client::num,
               +num_port: ring::server::num) {
    let mut num_chan <- some(num_chan);
    let mut num_port <- some(num_port);
    // Send/Receive lots of messages.
    for uint::range(0u, count) {|j|
        //#error("task %?, iter %?", i, j);
        let mut num_chan2 = none;
        let mut num_port2 = none;
        num_chan2 <-> num_chan;
        num_port2 <-> num_port;
        num_chan = some(ring::client::num(option::unwrap(num_chan2), i * j));
        let port = option::unwrap(num_port2);
        alt (#recv(port)) {
          ring::num(_n, p) {
            //log(error, _n);
            num_port = some(#move(p));
          }
        }
    };
}

fn main(args: [str]/~) {
    let args = if os::getenv("RUST_BENCH").is_some() {
        ["", "100", "10000"]/~
    } else if args.len() <= 1u {
        ["", "100", "1000"]/~
    } else {
        copy args
    }; 

    let num_tasks = option::get(uint::from_str(args[1]));
    let msg_per_task = option::get(uint::from_str(args[2]));

    let (num_chan, num_port) = ring::init();
    let mut num_chan = some(num_chan);

    let start = time::precise_time_s();

    // create the ring
    let mut futures = []/~;

    for uint::range(1u, num_tasks) {|i|
        //#error("spawning %?", i);
        let (new_chan, num_port) = ring::init();
        let num_chan2 = ~mut none;
        *num_chan2 <-> num_chan;
        let num_port = ~mut some(num_port);
        futures += [future::spawn {|move num_chan2, move num_port|
            let mut num_chan = none;
            num_chan <-> *num_chan2;
            let mut num_port1 = none;
            num_port1 <-> *num_port;
            thread_ring(i, msg_per_task,
                        option::unwrap(num_chan),
                        option::unwrap(num_port1))
        }]/~;
        num_chan = some(new_chan);
    };

    // do our iteration
    thread_ring(0u, msg_per_task, option::unwrap(num_chan), num_port);

    // synchronize
    for futures.each {|f| f.get() };

    let stop = time::precise_time_s();

    // all done, report stats.
    let num_msgs = num_tasks * msg_per_task;
    let elapsed = (stop - start);
    let rate = (num_msgs as float) / elapsed;

    io::println(#fmt("Sent %? messages in %? seconds",
                     num_msgs, elapsed));
    io::println(#fmt("  %? messages / second", rate));
    io::println(#fmt("  %? μs / message", 1000000. / rate));
}
