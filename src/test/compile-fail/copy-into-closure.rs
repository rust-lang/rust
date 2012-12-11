// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn closure1(+x: ~str) -> (~str, fn@() -> ~str) {
    let f = fn@() -> ~str {
        copy x
        //~^ WARNING implicitly copying a non-implicitly-copyable value
        //~^^ NOTE to copy values into a @fn closure, use a capture clause
    };
    (move x,f)
}

fn closure2(+x: util::NonCopyable) -> (util::NonCopyable,
                                       fn@() -> util::NonCopyable) {
    let f = fn@() -> util::NonCopyable {
        copy x
        //~^ ERROR copying a noncopyable value
        //~^^ NOTE non-copyable value cannot be copied into a @fn closure
        //~^^^ ERROR copying a noncopyable value
    };
    (move x,f)
}
fn closure3(+x: util::NonCopyable) {
    do task::spawn {
        let s = copy x;
        //~^ ERROR copying a noncopyable value
        //~^^ NOTE non-copyable value cannot be copied into a ~fn closure
        //~^^^ ERROR copying a noncopyable value
        error!("%?", s);
    }
    error!("%?", x);
}
fn main() {
    let x = ~"hello";
    do task::spawn {
        let s = copy x;
        //~^ WARNING implicitly copying a non-implicitly-copyable value
        //~^^ NOTE to copy values into a ~fn closure, use a capture clause
        error!("%s from child", s);
    }
    error!("%s", x);
}
