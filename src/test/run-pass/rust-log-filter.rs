// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// exec-env:RUST_LOG=rust-log-filter/f.o

#![feature(phase)]
#[phase(plugin,link)]
extern crate log;

pub struct ChannelLogger {
    tx: Sender<String>
}

impl ChannelLogger {
    pub fn new() -> (Box<ChannelLogger>, Receiver<String>) {
        let (tx, rx) = channel();
        (box ChannelLogger { tx: tx }, rx)
    }
}

impl log::Logger for ChannelLogger {
    fn log(&mut self, record: &log::LogRecord) {
        self.tx.send(format!("{}", record.args));
    }
}

pub fn main() {
    let (logger, rx) = ChannelLogger::new();

    spawn(proc() {
        log::set_logger(logger);

        // our regex is "f.o"
        // ensure it is a regex, and isn't anchored
        info!("foo");
        info!("bar");
        info!("foo bar");
        info!("bar foo");
        info!("f1o");
    });

    assert_eq!(rx.recv().as_slice(), "foo");
    assert_eq!(rx.recv().as_slice(), "foo bar");
    assert_eq!(rx.recv().as_slice(), "bar foo");
    assert_eq!(rx.recv().as_slice(), "f1o");
    assert!(rx.recv_opt().is_err());
}
