// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-cloudabi no std::net support

use std::net::TcpListener;
use std::net::TcpStream;
use std::io::{self, Read, Write};

fn handle_client(stream: TcpStream) -> io::Result<()> {
    stream.write_fmt(format!("message received"))
}

fn main() {
    if let Ok(listener) = TcpListener::bind("127.0.0.1:8080") {
        for incoming in listener.incoming() {
            if let Ok(stream) = incoming {
                handle_client(stream);
            }
        }
    }
}
