// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum msg { closed, received(~[u8]), }

fn producer(c: core::comm::Chan<~[u8]>) {
    core::comm::send(c, ~[1u8, 2u8, 3u8, 4u8]);
    let empty: ~[u8] = ~[];
    core::comm::send(c, empty);
}

fn packager(cb: core::comm::Chan<core::comm::Chan<~[u8]>>, msg: core::comm::Chan<msg>) {
    let p: core::comm::Port<~[u8]> = core::comm::Port();
    core::comm::send(cb, core::comm::Chan(&p));
    loop {
        debug!("waiting for bytes");
        let data = core::comm::recv(p);
        debug!("got bytes");
        if vec::len(data) == 0u {
            debug!("got empty bytes, quitting");
            break;
        }
        debug!("sending non-empty buffer of length");
        log(debug, vec::len(data));
        core::comm::send(msg, received(data));
        debug!("sent non-empty buffer");
    }
    debug!("sending closed message");
    core::comm::send(msg, closed);
    debug!("sent closed message");
}

fn main() {
    let p: core::comm::Port<msg> = core::comm::Port();
    let ch = core::comm::Chan(&p);
    let recv_reader: core::comm::Port<core::comm::Chan<~[u8]>> = core::comm::Port();
    let recv_reader_chan = core::comm::Chan(&recv_reader);
    let pack = task::spawn(|| packager(recv_reader_chan, ch) );

    let source_chan: core::comm::Chan<~[u8]> = core::comm::recv(recv_reader);
    let prod = task::spawn(|| producer(source_chan) );

    loop {
        let msg = core::comm::recv(p);
        match msg {
          closed => { debug!("Got close message"); break; }
          received(data) => {
            debug!("Got data. Length is:");
            log(debug, vec::len::<u8>(data));
          }
        }
    }
}
