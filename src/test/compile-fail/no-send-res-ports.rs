// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unsafe_destructor)]

use std::thread::Thread;
use std::rc::Rc;

#[derive(Show)]
struct Port<T>(Rc<T>);

fn main() {
    #[derive(Show)]
    struct foo {
      _x: Port<()>,
    }

    #[unsafe_destructor]
    impl Drop for foo {
        fn drop(&mut self) {}
    }

    fn foo(x: Port<()>) -> foo {
        foo {
            _x: x
        }
    }

    let x = foo(Port(Rc::new(())));

    Thread::spawn(move|| {
        //~^ ERROR `core::marker::Send` is not implemented
        //~^^ ERROR `core::marker::Send` is not implemented
        let y = x;
        println!("{:?}", y);
    });
}
