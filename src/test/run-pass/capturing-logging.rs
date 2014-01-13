// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast
// xfail-android (FIXME #11419)
// exec-env:RUST_LOG=info

#[no_uv];
extern mod native;

use std::fmt;
use std::io::comm_adapters::{PortReader, ChanWriter};
use std::logging::{set_logger, Logger};

struct MyWriter(ChanWriter);

impl Logger for MyWriter {
    fn log(&mut self, _level: u32, args: &fmt::Arguments) {
        let MyWriter(ref mut inner) = *self;
        fmt::writeln(inner as &mut Writer, args);
    }
}

#[start]
fn start(argc: int, argv: **u8) -> int {
    do native::start(argc, argv) {
        main();
    }
}

fn main() {
    let (p, c) = Chan::new();
    let (mut r, w) = (PortReader::new(p), ChanWriter::new(c));
    do spawn {
        set_logger(~MyWriter(w) as ~Logger);
        debug!("debug");
        info!("info");
    }
    assert_eq!(r.read_to_str(), ~"info\n");
}
