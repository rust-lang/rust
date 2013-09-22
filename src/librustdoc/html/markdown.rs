// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;
use std::rt::io::Reader;
use std::rt::io::pipe::PipeStream;
use std::rt::io::process::{ProcessConfig, Process, CreatePipe};
use std::rt::io;

pub struct Markdown<'self>(&'self str);

impl<'self> fmt::Default for Markdown<'self> {
    fn fmt(md: &Markdown<'self>, fmt: &mut fmt::Formatter) {
        if md.len() == 0 { return; }

        // Create the pandoc process
        do io::io_error::cond.trap(|err| {
            fail2!("Error executing `pandoc`: {}", err.desc);
        }).inside {
            let io = ~[CreatePipe(PipeStream::new().unwrap(), true, false),
                       CreatePipe(PipeStream::new().unwrap(), false, true)];
            let args = ProcessConfig {
                program: "pandoc",
                args: [],
                env: None,
                cwd: None,
                io: io,
            };
            let mut p = Process::new(args).expect("couldn't fork for pandoc");

            // Write the markdown to stdin and close it.
            p.io[0].get_mut_ref().write(md.as_bytes());
            p.io[0] = None;

            // Ferry the output from pandoc over to the destination buffer.
            let mut buf = [0, ..1024];
            loop {
                match p.io[1].get_mut_ref().read(buf) {
                    None | Some(0) => { break }
                    Some(n) => {
                        fmt.buf.write(buf.slice_to(n));
                    }
                }
            }
        }
    }
}
