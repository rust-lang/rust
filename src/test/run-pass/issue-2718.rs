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
    mod rusti {
      fn atomic_xchng(&dst: int, src: int) -> int { fail; }
      fn atomic_xchng_acq(&dst: int, src: int) -> int { fail; }
      fn atomic_xchng_rel(&dst: int, src: int) -> int { fail; }
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
          full { fail ~"duplicate send" }
          blocked {

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
                return some(option::unwrap(payload))
              }
              terminated {
                assert old_state == terminated;
                return none;
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
            fail ~"you dun goofed"
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
            fail ~"terminating a blocked packet"
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
    enum ping = pipes::send_packet<pong>;
    enum pong = pipes::send_packet<ping>;

    fn liberate_ping(-p: ping) -> pipes::send_packet<pong> unsafe {
        let addr : *pipes::send_packet<pong> = alt p {
          ping(x) { unsafe::reinterpret_cast(ptr::addr_of(x)) }
        };
        let liberated_value <- *addr;
        unsafe::forget(p);
        liberated_value
    }

    fn liberate_pong(-p: pong) -> pipes::send_packet<ping> unsafe {
        let addr : *pipes::send_packet<ping> = alt p {
          pong(x) { unsafe::reinterpret_cast(ptr::addr_of(x)) }
        };
        let liberated_value <- *addr;
        unsafe::forget(p);
        liberated_value
    }

    fn init() -> (client::ping, server::ping) {
        pipes::entangle()
    }

    mod client {
        type ping = pipes::send_packet<pingpong::ping>;
        type pong = pipes::recv_packet<pingpong::pong>;

        fn do_ping(-c: ping) -> pong {
            let (sp, rp) = pipes::entangle();

            pipes::send(c, ping(sp));
            rp
        }

        fn do_pong(-c: pong) -> (ping, ()) {
            let packet = pipes::recv(c);
            if packet == none {
                fail ~"sender closed the connection"
            }
            (liberate_pong(option::unwrap(packet)), ())
        }
    }

    mod server {
        type ping = pipes::recv_packet<pingpong::ping>;
        type pong = pipes::send_packet<pingpong::pong>;

        fn do_ping(-c: ping) -> (pong, ()) {
            let packet = pipes::recv(c);
            if packet == none {
                fail ~"sender closed the connection"
            }
            (liberate_ping(option::unwrap(packet)), ())
        }

        fn do_pong(-c: pong) -> ping {
            let (sp, rp) = pipes::entangle();
            pipes::send(c, pong(sp));
            rp
        }
    }
}

fn client(-chan: pingpong::client::ping) {
    let chan = pingpong::client::do_ping(chan);
    log(error, ~"Sent ping");
    let (chan, _data) = pingpong::client::do_pong(chan);
    log(error, ~"Received pong");
}

fn server(-chan: pingpong::server::ping) {
    let (chan, _data) = pingpong::server::do_ping(chan);
    log(error, ~"Received ping");
    let chan = pingpong::server::do_pong(chan);
    log(error, ~"Sent pong");
}

fn main() {
  /*
//    Commented out because of option::get error

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
  */
}
