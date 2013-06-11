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

extern mod extra;

use extra::timer::sleep;
use extra::uv;
use std::cell::Cell;
use std::pipes::*;
use std::pipes;
use std::task;

proto! oneshot (
    waiting:send {
        signal -> !
    }
)


/** Spawn a task to provide a service.

It takes an initialization function that produces a send and receive
endpoint. The send endpoint is returned to the caller and the receive
endpoint is passed to the new task.

*/
pub fn spawn_service<T:Send,Tb:Send>(
            init: extern fn() -> (RecvPacketBuffered<T, Tb>,
                                  SendPacketBuffered<T, Tb>),
            service: ~fn(v: RecvPacketBuffered<T, Tb>))
        -> SendPacketBuffered<T, Tb> {
    let (server, client) = init();

    // This is some nasty gymnastics required to safely move the pipe
    // into a new task.
    let server = Cell::new(server);
    do task::spawn {
        service(server.take());
    }

    client
}

pub fn main() {
    use oneshot::client::*;

    let c = spawn_service(oneshot::init, |p| { recv(p); });

    let iotask = &uv::global_loop::get();
    sleep(iotask, 500);

    signal(c);
}
