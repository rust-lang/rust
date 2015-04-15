// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// exec-env:RUST_LOG=rust_log_filter/foo


#![allow(unknown_features)]
#![feature(box_syntax, std_misc, rustc_private)]

#[macro_use]
extern crate log;

use std::sync::mpsc::{channel, Sender, Receiver};
use std::thread;

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
        self.tx.send(format!("{}", record.args)).unwrap();
    }
}

pub fn main() {
    let (logger, rx) = ChannelLogger::new();

    let t = thread::spawn(move|| {
        log::set_logger(logger);

        info!("foo");
        info!("bar");
        info!("foo bar");
        info!("bar foo");
    });

    assert_eq!(rx.recv().unwrap(), "foo");
    assert_eq!(rx.recv().unwrap(), "foo bar");
    assert_eq!(rx.recv().unwrap(), "bar foo");
    assert!(rx.recv().is_err());

    t.join();
}
