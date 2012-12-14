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

fn producer(c: core::oldcomm::Chan<~[u8]>) {
    core::oldcomm::send(c, ~[1u8, 2u8, 3u8, 4u8]);
    let empty: ~[u8] = ~[];
    core::oldcomm::send(c, empty);
}

fn packager(cb: core::oldcomm::Chan<core::oldcomm::Chan<~[u8]>>, msg: core::oldcomm::Chan<msg>) {
    let p: core::oldcomm::Port<~[u8]> = core::oldcomm::Port();
    core::oldcomm::send(cb, core::oldcomm::Chan(&p));
    loop {
        debug!("waiting for bytes");
        let data = core::oldcomm::recv(p);
        debug!("got bytes");
        if vec::len(data) == 0u {
            debug!("got empty bytes, quitting");
            break;
        }
        debug!("sending non-empty buffer of length");
        log(debug, vec::len(data));
        core::oldcomm::send(msg, received(data));
        debug!("sent non-empty buffer");
    }
    debug!("sending closed message");
    core::oldcomm::send(msg, closed);
    debug!("sent closed message");
}

fn main() {
    let p: core::oldcomm::Port<msg> = core::oldcomm::Port();
    let ch = core::oldcomm::Chan(&p);
    let recv_reader: core::oldcomm::Port<core::oldcomm::Chan<~[u8]>> = core::oldcomm::Port();
    let recv_reader_chan = core::oldcomm::Chan(&recv_reader);
    let pack = task::spawn(|| packager(recv_reader_chan, ch) );

    let source_chan: core::oldcomm::Chan<~[u8]> = core::oldcomm::recv(recv_reader);
    let prod = task::spawn(|| producer(source_chan) );

    loop {
        let msg = core::oldcomm::recv(p);
        match msg {
          closed => { debug!("Got close message"); break; }
          received(data) => {
            debug!("Got data. Length is:");
            log(debug, vec::len::<u8>(data));
          }
        }
    }
}
