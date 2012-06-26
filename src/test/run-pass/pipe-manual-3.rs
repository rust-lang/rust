/*
The first test case using pipes. The idea is to break this into
several stages for prototyping. Here's the plan:

1. Write an already-compiled protocol using existing ports and chans.

2. Take the already-compiled version and add the low-level
synchronization code instead.

3. Write a syntax extension to compile the protocols.

At some point, we'll need to add support for select.

This file does horrible things to pretend we have self-move.

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
        new(p: *packet<T>) { self.p = some(p); }
        drop {
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
        new(p: *packet<T>) { self.p = some(p); }
        drop {
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

mod pingpong {
    enum ping { ping, }
    enum ping_message = *pipes::packet<pong_message>;
    enum pong { pong, }
    enum pong_message = *pipes::packet<ping_message>;

    fn init() -> (client::ping, server::ping) {
        pipes::entangle()
    }

    mod client {
        type ping = pipes::send_packet<pingpong::ping_message>;
        type pong = pipes::recv_packet<pingpong::pong_message>;
    }

    impl abominable for client::ping {
        fn send() -> fn@(-client::ping, ping) -> client::pong {
            {|pipe, data|
                let p = pipes::packet();
                pipes::send(pipe, pingpong::ping_message(p));
                pipes::recv_packet(p)
            }
        }
    }

    impl abominable for client::pong {
        fn recv() -> fn@(-client::pong) -> (client::ping, pong) {
            {|pipe|
                let packet = pipes::recv(pipe);
                if packet == none {
                    fail "sender closed the connection"
                }
                let p : pong_message = option::unwrap(packet);
                (pipes::send_packet(*p), pong)
            }
        }
    }

    mod server {
        type ping = pipes::recv_packet<pingpong::ping_message>;
        type pong = pipes::send_packet<pingpong::pong_message>;
    }

    impl abominable for server::ping {
        fn recv() -> fn@(-server::ping) -> (server::pong, ping) {
            {|pipe|
                let packet = pipes::recv(pipe);
                if packet == none {
                    fail "sender closed the connection"
                }
                let p : ping_message = option::unwrap(packet);
                (pipes::send_packet(*p), ping)
            }
        }
    }

    impl abominable for server::pong {
        fn send() -> fn@(-server::pong, pong) -> server::ping {
            {|pipe, data|
                let p = pipes::packet();
                pipes::send(pipe, pingpong::pong_message(p));
                pipes::recv_packet(p)
            }
        }
    }
}

mod test {
    import pingpong::{ping, pong, abominable};

    fn macros() {
        #macro[
            [#send[chan, data, ...],
             chan.send()(chan, data, ...)]
        ];
        #macro[
            [#recv[chan],
             chan.recv()(chan)]
        ];
    }

    fn client(-chan: pingpong::client::ping) {
        let chan = #send(chan, ping);
        log(error, "Sent ping");
        let (chan, _data) = #recv(chan);
        log(error, "Received pong");
    }
    
    fn server(-chan: pingpong::server::ping) {
        let (chan, _data) = #recv(chan);
        log(error, "Received ping");
        let chan = #send(chan, pong);
        log(error, "Sent pong");
    }
}

fn main() {
    let (client_, server_) = pingpong::init();
    let client_ = ~mut some(client_);
    let server_ = ~mut some(server_);

    task::spawn {|move client_|
        let mut client__ = none;
        *client_ <-> client__;
        test::client(option::unwrap(client__));
    };
    task::spawn {|move server_|
        let mut server_ˊ = none;
        *server_ <-> server_ˊ;
        test::server(option::unwrap(server_ˊ));
    };
}
