// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]

trait Resources {
    type Buffer: Copy;
}

#[derive(Copy, Clone)]
struct ConstantBufferSet<R: Resources>(
    pub R::Buffer
);

#[derive(Copy, Clone)]
enum It {}
impl Resources for It {
    type Buffer = u8;
}

#[derive(Copy, Clone)]
enum Command {
    BindConstantBuffers(ConstantBufferSet<It>)
}

fn main() {}
