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

// xfail-pretty
// xfail-win32

extern mod std;
use std::timer::sleep;
use std::uv;

use core::pipes;
use core::pipes::{recv, select};

proto! oneshot (
    waiting:send {
        signal -> !
    }
)

proto! stream (
    Stream:send<T:Owned> {
        send(T) -> Stream<T>
    }
)

pub fn main() {
    use oneshot::client::*;
    use stream::client::*;

    let iotask = &uv::global_loop::get();
    
    let c = pipes::spawn_service(stream::init, |p| { 
        error!("waiting for pipes");
        let stream::send(x, p) = recv(p);
        error!("got pipes");
        let (left, right) : (oneshot::server::waiting,
                             oneshot::server::waiting)
            = x;
        error!("selecting");
        let (i, _, _) = select(~[left, right]);
        error!("selected");
        assert!(i == 0);

        error!("waiting for pipes");
        let stream::send(x, _) = recv(p);
        error!("got pipes");
        let (left, right) : (oneshot::server::waiting,
                             oneshot::server::waiting)
            = x;
        error!("selecting");
        let (i, m, _) = select(~[left, right]);
        error!("selected %?", i);
        if m.is_some() {
            assert!(i == 1);
        }
    });

    let (c1, p1) = oneshot::init();
    let (_c2, p2) = oneshot::init();

    let c = send(c, (p1, p2));
    
    sleep(iotask, 100);

    signal(c1);

    let (_c1, p1) = oneshot::init();
    let (c2, p2) = oneshot::init();

    send(c, (p1, p2));

    sleep(iotask, 100);

    signal(c2);

    test_select2();
}

fn test_select2() {
    let (ac, ap) = stream::init();
    let (bc, bp) = stream::init();

    stream::client::send(ac, 42);

    match pipes::select2(ap, bp) {
      either::Left(*) => { }
      either::Right(*) => { fail!() }
    }

    stream::client::send(bc, ~"abc");

    error!("done with first select2");

    let (ac, ap) = stream::init();
    let (bc, bp) = stream::init();

    stream::client::send(bc, ~"abc");

    match pipes::select2(ap, bp) {
      either::Left(*) => { fail!() }
      either::Right(*) => { }
    }

    stream::client::send(ac, 42);
}
