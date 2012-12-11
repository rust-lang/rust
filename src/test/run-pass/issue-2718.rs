// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// tjc: I don't know why
mod pipes {
    #[legacy_exports];
    use cast::{forget, transmute};

    enum state {
        empty,
        full,
        blocked,
        terminated
    }

    impl state : cmp::Eq {
        pure fn eq(&self, other: &state) -> bool {
            ((*self) as uint) == ((*other) as uint)
        }
        pure fn ne(&self, other: &state) -> bool { !(*self).eq(other) }
    }

    type packet<T: Send> = {
        mut state: state,
        mut blocked_task: Option<task::Task>,
        mut payload: Option<T>
    };

    fn packet<T: Send>() -> *packet<T> unsafe {
        let p: *packet<T> = cast::transmute(~{
            mut state: empty,
            mut blocked_task: None::<task::Task>,
            mut payload: None::<T>
        });
        p
    }

    #[abi = "rust-intrinsic"]
    mod rusti {
        #[legacy_exports];
      fn atomic_xchg(_dst: &mut int, _src: int) -> int { fail; }
      fn atomic_xchg_acq(_dst: &mut int, _src: int) -> int { fail; }
      fn atomic_xchg_rel(_dst: &mut int, _src: int) -> int { fail; }
    }

    // We should consider moving this to core::unsafe, although I
    // suspect graydon would want us to use void pointers instead.
    unsafe fn uniquify<T>(+x: *T) -> ~T {
        unsafe { cast::transmute(move x) }
    }

    fn swap_state_acq(+dst: &mut state, src: state) -> state {
        unsafe {
            transmute(rusti::atomic_xchg_acq(transmute(move dst), src as int))
        }
    }

    fn swap_state_rel(+dst: &mut state, src: state) -> state {
        unsafe {
            transmute(rusti::atomic_xchg_rel(transmute(move dst), src as int))
        }
    }

    fn send<T: Send>(-p: send_packet<T>, -payload: T) {
        let p = p.unwrap();
        let p = unsafe { uniquify(p) };
        assert (*p).payload.is_none();
        (*p).payload = move Some(move payload);
        let old_state = swap_state_rel(&mut (*p).state, full);
        match old_state {
          empty => {
            // Yay, fastpath.

            // The receiver will eventually clean this up.
            unsafe { forget(move p); }
          }
          full => { fail ~"duplicate send" }
          blocked => {

            // The receiver will eventually clean this up.
            unsafe { forget(move p); }
          }
          terminated => {
            // The receiver will never receive this. Rely on drop_glue
            // to clean everything up.
          }
        }
    }

    fn recv<T: Send>(-p: recv_packet<T>) -> Option<T> {
        let p = p.unwrap();
        let p = unsafe { uniquify(p) };
        loop {
            let old_state = swap_state_acq(&mut (*p).state,
                                           blocked);
            match old_state {
              empty | blocked => { task::yield(); }
              full => {
                let mut payload = None;
                payload <-> (*p).payload;
                return Some(option::unwrap(move payload))
              }
              terminated => {
                assert old_state == terminated;
                return None;
              }
            }
        }
    }

    fn sender_terminate<T: Send>(p: *packet<T>) {
        let p = unsafe { uniquify(p) };
        match swap_state_rel(&mut (*p).state, terminated) {
          empty | blocked => {
            // The receiver will eventually clean up.
            unsafe { forget(move p) }
          }
          full => {
            // This is impossible
            fail ~"you dun goofed"
          }
          terminated => {
            // I have to clean up, use drop_glue
          }
        }
    }

    fn receiver_terminate<T: Send>(p: *packet<T>) {
        let p = unsafe { uniquify(p) };
        match swap_state_rel(&mut (*p).state, terminated) {
          empty => {
            // the sender will clean up
            unsafe { forget(move p) }
          }
          blocked => {
            // this shouldn't happen.
            fail ~"terminating a blocked packet"
          }
          terminated | full => {
            // I have to clean up, use drop_glue
          }
        }
    }

    struct send_packet<T: Send> {
        mut p: Option<*packet<T>>,
    }

    impl<T: Send> send_packet<T> : Drop {
        fn finalize(&self) {
            if self.p != None {
                let mut p = None;
                p <-> self.p;
                sender_terminate(option::unwrap(move p))
            }
        }
    }

    impl<T: Send> send_packet<T> {
        fn unwrap() -> *packet<T> {
            let mut p = None;
            p <-> self.p;
            option::unwrap(move p)
        }
    }

    fn send_packet<T: Send>(p: *packet<T>) -> send_packet<T> {
        send_packet {
            p: Some(p)
        }
    }

    struct recv_packet<T: Send> {
        mut p: Option<*packet<T>>,
    }

    impl<T: Send> recv_packet<T> : Drop {
        fn finalize(&self) {
            if self.p != None {
                let mut p = None;
                p <-> self.p;
                receiver_terminate(option::unwrap(move p))
            }
        }
    }

    impl<T: Send> recv_packet<T> {
        fn unwrap() -> *packet<T> {
            let mut p = None;
            p <-> self.p;
            option::unwrap(move p)
        }
    }

    fn recv_packet<T: Send>(p: *packet<T>) -> recv_packet<T> {
        recv_packet {
            p: Some(p)
        }
    }

    fn entangle<T: Send>() -> (send_packet<T>, recv_packet<T>) {
        let p = packet();
        (send_packet(p), recv_packet(p))
    }
}

mod pingpong {
    #[legacy_exports];
    enum ping = pipes::send_packet<pong>;
    enum pong = pipes::send_packet<ping>;

    fn liberate_ping(-p: ping) -> pipes::send_packet<pong> unsafe {
        let addr : *pipes::send_packet<pong> = match &p {
          &ping(x) => { cast::transmute(ptr::addr_of(&x)) }
        };
        let liberated_value = move *addr;
        cast::forget(move p);
        move liberated_value
    }

    fn liberate_pong(-p: pong) -> pipes::send_packet<ping> unsafe {
        let addr : *pipes::send_packet<ping> = match &p {
          &pong(x) => { cast::transmute(ptr::addr_of(&x)) }
        };
        let liberated_value = move *addr;
        cast::forget(move p);
        move liberated_value
    }

    fn init() -> (client::ping, server::ping) {
        pipes::entangle()
    }

    mod client {
        #[legacy_exports];
        type ping = pipes::send_packet<pingpong::ping>;
        type pong = pipes::recv_packet<pingpong::pong>;

        fn do_ping(-c: ping) -> pong {
            let (sp, rp) = pipes::entangle();

            pipes::send(move c, ping(move sp));
            move rp
        }

        fn do_pong(-c: pong) -> (ping, ()) {
            let packet = pipes::recv(move c);
            if packet.is_none() {
                fail ~"sender closed the connection"
            }
            (liberate_pong(option::unwrap(move packet)), ())
        }
    }

    mod server {
        #[legacy_exports];
        type ping = pipes::recv_packet<pingpong::ping>;
        type pong = pipes::send_packet<pingpong::pong>;

        fn do_ping(-c: ping) -> (pong, ()) {
            let packet = pipes::recv(move c);
            if packet.is_none() {
                fail ~"sender closed the connection"
            }
            (liberate_ping(option::unwrap(move packet)), ())
        }

        fn do_pong(-c: pong) -> ping {
            let (sp, rp) = pipes::entangle();
            pipes::send(move c, pong(move sp));
            move rp
        }
    }
}

fn client(-chan: pingpong::client::ping) {
    let chan = pingpong::client::do_ping(move chan);
    log(error, ~"Sent ping");
    let (_chan, _data) = pingpong::client::do_pong(move chan);
    log(error, ~"Received pong");
}

fn server(-chan: pingpong::server::ping) {
    let (chan, _data) = pingpong::server::do_ping(move chan);
    log(error, ~"Received ping");
    let _chan = pingpong::server::do_pong(move chan);
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
