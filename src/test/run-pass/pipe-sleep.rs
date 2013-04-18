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

extern mod std;
use std::timer::sleep;
use std::uv;
use core::cell::Cell;
use core::pipes;
use core::pipes::*;

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
pub fn spawn_service<T:Owned,Tb:Owned>(
            init: extern fn() -> (SendPacketBuffered<T, Tb>,
                                  RecvPacketBuffered<T, Tb>),
            service: ~fn(v: RecvPacketBuffered<T, Tb>))
        -> SendPacketBuffered<T, Tb> {
    let (client, server) = init();

    // This is some nasty gymnastics required to safely move the pipe
    // into a new task.
    let server = Cell(server);
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
