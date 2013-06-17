// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub mod stream {
    pub enum Stream<T:Owned> { send(T, ::stream::server::Stream<T>), }
    pub mod server {
        use std::option;
        use std::pipes;

        impl<T:Owned> Stream<T> {
            pub fn recv() -> extern fn(v: Stream<T>) -> ::stream::Stream<T> {
              // resolve really should report just one error here.
              // Change the test case when it changes.
              pub fn recv(pipe: Stream<T>) -> ::stream::Stream<T> { //~ ERROR attempt to use a type argument out of scope
                //~^ ERROR use of undeclared type name
                //~^^ ERROR attempt to use a type argument out of scope
                //~^^^ ERROR use of undeclared type name
                    pipes::recv(pipe).unwrap()
                }
                recv
            }
        }

        pub type Stream<T:Owned> = pipes::RecvPacket<::stream::Stream<T>>;
    }
}

fn main() {}
