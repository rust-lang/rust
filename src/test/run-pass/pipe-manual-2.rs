/*
The first test case using pipes. The idea is to break this into
several stages for prototyping. Here's the plan:

1. Write an already-compiled protocol using existing ports and chans.

2. Take the already-compiled version and add the low-level
synchronization code instead. (That's what this file attempts to do)

3. Write a syntax extension to compile the protocols.

At some point, we'll need to add support for select.

*/

// Hopefully someday we'll move this into core.
mod pipes {
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

    fn send<T: send>(p: *packet<T>, -payload: T) {
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

    fn recv<T: send>(p: *packet<T>) -> option<T> {
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
}

mod pingpong {
    enum ping = *pipes::packet<pong>;
    enum pong = *pipes::packet<ping>;

    fn init() -> (client::ping, server::ping) {
        let p = pipes::packet();
        let p = pingpong::ping(p);

        let client = client::ping(p);
        let server = server::ping(p);

        (client, server)
    }

    mod client {
        enum ping = pingpong::ping;
        enum pong = pingpong::pong;

        fn do_ping(-c: ping) -> pong {
            let packet = pipes::packet();
            let packet = pingpong::pong(packet);

            pipes::send(**c, copy packet);
            pong(packet)
        }

        fn do_pong(-c: pong) -> (ping, ()) {
            let packet = pipes::recv(**c);
            alt packet {
              none {
                fail "sender closed the connection"
              }
              some(new_packet) {
                (ping(new_packet), ())
              }
            }
        }
    }

    mod server {
        enum ping = pingpong::ping;
        enum pong = pingpong::pong;

        fn do_ping(-c: ping) -> (pong, ()) {
            let packet = pipes::recv(**c);
            alt packet {
              none { fail "sender closed the connection" }
              some(new_packet) {
                (pong(new_packet), ())
              }
            }
        }

        fn do_pong(-c: pong) -> ping {
            let packet = pipes::packet();
            let packet = pingpong::ping(packet);

            pipes::send(**c, copy packet);
            ping(packet)
        }
    }
}

fn client(-chan: pingpong::client::ping) {
    let chan = pingpong::client::do_ping(chan);
    log(error, "Sent ping");
    let (chan, _data) = pingpong::client::do_pong(chan);
    log(error, "Received pong");
    pipes::sender_terminate(**chan);
}

fn server(-chan: pingpong::server::ping) {
    let (chan, _data) = pingpong::server::do_ping(chan);
    log(error, "Received ping");
    let chan = pingpong::server::do_pong(chan);
    log(error, "Sent pong");
    pipes::receiver_terminate(**chan);
}

fn main() {
    let (client_, server_) = pingpong::init();
    let client_ = ~mut some(client_);
    let server_ = ~mut some(server_);

    task::spawn {|move client_|
        let mut client__ = none;
        *client_ <-> client__;
        client(option::unwrap(client__));
    };
    task::spawn {|move server_|
        let mut server_ˊ = none;
        *server_ <-> server_ˊ;
        server(option::unwrap(server_ˊ));
    };
}
