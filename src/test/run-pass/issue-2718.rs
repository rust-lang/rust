// xfail-fast

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub type Task = int;

// tjc: I don't know why
pub mod pipes {
    use super::Task;
    use std::cast::{forget, transmute};
    use std::cast;
    use std::task;
    use std::util;

    pub struct Stuff<T> {
        state: state,
        blocked_task: Option<Task>,
        payload: Option<T>
    }

    #[deriving(Eq)]
    pub enum state {
        empty,
        full,
        blocked,
        terminated
    }

    pub struct packet<T> {
        state: state,
        blocked_task: Option<Task>,
        payload: Option<T>
    }

    pub fn packet<T:Send>() -> *packet<T> {
        unsafe {
            let p: *packet<T> = cast::transmute(~Stuff{
                state: empty,
                blocked_task: None::<Task>,
                payload: None::<T>
            });
            p
        }
    }

    mod rusti {
      pub fn atomic_xchg(_dst: &mut int, _src: int) -> int { fail!(); }
      pub fn atomic_xchg_acq(_dst: &mut int, _src: int) -> int { fail!(); }
      pub fn atomic_xchg_rel(_dst: &mut int, _src: int) -> int { fail!(); }
    }

    // We should consider moving this to ::std::unsafe, although I
    // suspect graydon would want us to use void pointers instead.
    pub unsafe fn uniquify<T>(x: *T) -> ~T {
        cast::transmute(x)
    }

    pub fn swap_state_acq(dst: &mut state, src: state) -> state {
        unsafe {
            transmute(rusti::atomic_xchg_acq(transmute(dst), src as int))
        }
    }

    pub fn swap_state_rel(dst: &mut state, src: state) -> state {
        unsafe {
            transmute(rusti::atomic_xchg_rel(transmute(dst), src as int))
        }
    }

    pub fn send<T:Send>(mut p: send_packet<T>, payload: T) {
        let p = p.unwrap();
        let mut p = unsafe { uniquify(p) };
        assert!((*p).payload.is_none());
        (*p).payload = Some(payload);
        let old_state = swap_state_rel(&mut (*p).state, full);
        match old_state {
          empty => {
            // Yay, fastpath.

            // The receiver will eventually clean this up.
            unsafe { forget(p); }
          }
          full => { fail!("duplicate send") }
          blocked => {

            // The receiver will eventually clean this up.
            unsafe { forget(p); }
          }
          terminated => {
            // The receiver will never receive this. Rely on drop_glue
            // to clean everything up.
          }
        }
    }

    pub fn recv<T:Send>(mut p: recv_packet<T>) -> Option<T> {
        let p = p.unwrap();
        let mut p = unsafe { uniquify(p) };
        loop {
            let old_state = swap_state_acq(&mut (*p).state,
                                           blocked);
            match old_state {
              empty | blocked => { task::deschedule(); }
              full => {
                let payload = util::replace(&mut p.payload, None);
                return Some(payload.unwrap())
              }
              terminated => {
                assert_eq!(old_state, terminated);
                return None;
              }
            }
        }
    }

    pub fn sender_terminate<T:Send>(p: *packet<T>) {
        let mut p = unsafe { uniquify(p) };
        match swap_state_rel(&mut (*p).state, terminated) {
          empty | blocked => {
            // The receiver will eventually clean up.
            unsafe { forget(p) }
          }
          full => {
            // This is impossible
            fail!("you dun goofed")
          }
          terminated => {
            // I have to clean up, use drop_glue
          }
        }
    }

    pub fn receiver_terminate<T:Send>(p: *packet<T>) {
        let mut p = unsafe { uniquify(p) };
        match swap_state_rel(&mut (*p).state, terminated) {
          empty => {
            // the sender will clean up
            unsafe { forget(p) }
          }
          blocked => {
            // this shouldn't happen.
            fail!("terminating a blocked packet")
          }
          terminated | full => {
            // I have to clean up, use drop_glue
          }
        }
    }

    pub struct send_packet<T> {
        p: Option<*packet<T>>,
    }

    #[unsafe_destructor]
    impl<T:Send> Drop for send_packet<T> {
        fn drop(&mut self) {
            unsafe {
                if self.p != None {
                    let self_p: &mut Option<*packet<T>> =
                        cast::transmute(&self.p);
                    let p = util::replace(self_p, None);
                    sender_terminate(p.unwrap())
                }
            }
        }
    }

    impl<T:Send> send_packet<T> {
        pub fn unwrap(&mut self) -> *packet<T> {
            util::replace(&mut self.p, None).unwrap()
        }
    }

    pub fn send_packet<T:Send>(p: *packet<T>) -> send_packet<T> {
        send_packet {
            p: Some(p)
        }
    }

    pub struct recv_packet<T> {
        p: Option<*packet<T>>,
    }

    #[unsafe_destructor]
    impl<T:Send> Drop for recv_packet<T> {
        fn drop(&mut self) {
            unsafe {
                if self.p != None {
                    let self_p: &mut Option<*packet<T>> =
                        cast::transmute(&self.p);
                    let p = util::replace(self_p, None);
                    receiver_terminate(p.unwrap())
                }
            }
        }
    }

    impl<T:Send> recv_packet<T> {
        pub fn unwrap(&mut self) -> *packet<T> {
            util::replace(&mut self.p, None).unwrap()
        }
    }

    pub fn recv_packet<T:Send>(p: *packet<T>) -> recv_packet<T> {
        recv_packet {
            p: Some(p)
        }
    }

    pub fn entangle<T:Send>() -> (send_packet<T>, recv_packet<T>) {
        let p = packet();
        (send_packet(p), recv_packet(p))
    }
}

pub mod pingpong {
    use std::cast;

    pub struct ping(::pipes::send_packet<pong>);
    pub struct pong(::pipes::send_packet<ping>);

    pub fn liberate_ping(p: ping) -> ::pipes::send_packet<pong> {
        unsafe {
            let _addr : *::pipes::send_packet<pong> = match &p {
              &ping(ref x) => { cast::transmute(x) }
            };
            fail!()
        }
    }

    pub fn liberate_pong(p: pong) -> ::pipes::send_packet<ping> {
        unsafe {
            let _addr : *::pipes::send_packet<ping> = match &p {
              &pong(ref x) => { cast::transmute(x) }
            };
            fail!()
        }
    }

    pub fn init() -> (client::ping, server::ping) {
        ::pipes::entangle()
    }

    pub mod client {
        use pingpong;

        pub type ping = ::pipes::send_packet<pingpong::ping>;
        pub type pong = ::pipes::recv_packet<pingpong::pong>;

        pub fn do_ping(c: ping) -> pong {
            let (sp, rp) = ::pipes::entangle();

            ::pipes::send(c, pingpong::ping(sp));
            rp
        }

        pub fn do_pong(c: pong) -> (ping, ()) {
            let packet = ::pipes::recv(c);
            if packet.is_none() {
                fail!("sender closed the connection")
            }
            (pingpong::liberate_pong(packet.unwrap()), ())
        }
    }

    pub mod server {
        use pingpong;

        pub type ping = ::pipes::recv_packet<pingpong::ping>;
        pub type pong = ::pipes::send_packet<pingpong::pong>;

        pub fn do_ping(c: ping) -> (pong, ()) {
            let packet = ::pipes::recv(c);
            if packet.is_none() {
                fail!("sender closed the connection")
            }
            (pingpong::liberate_ping(packet.unwrap()), ())
        }

        pub fn do_pong(c: pong) -> ping {
            let (sp, rp) = ::pipes::entangle();
            ::pipes::send(c, pingpong::pong(sp));
            rp
        }
    }
}

fn client(chan: pingpong::client::ping) {
    let chan = pingpong::client::do_ping(chan);
    error!("Sent ping");
    let (_chan, _data) = pingpong::client::do_pong(chan);
    error!("Received pong");
}

fn server(chan: pingpong::server::ping) {
    let (chan, _data) = pingpong::server::do_ping(chan);
    error!("Received ping");
    let _chan = pingpong::server::do_pong(chan);
    error!("Sent pong");
}

pub fn main() {
  /*
//    Commented out because of option::get error

    let (client_, server_) = pingpong::init();
    let client_ = Cell::new(client_);
    let server_ = Cell::new(server_);

    task::spawn {|client_|
        let client__ = client_.take();
        client(client__);
    };
    task::spawn {|server_|
        let server__ = server_.take();
        server(server_ˊ);
    };
  */
}
