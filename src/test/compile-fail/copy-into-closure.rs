// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn closure2(+x: util::NonCopyable) -> (util::NonCopyable,
                                       fn@() -> util::NonCopyable) {
    let f = fn@(copy x) -> util::NonCopyable {
        //~^ ERROR copying a noncopyable value
        //~^^ NOTE non-copyable value cannot be copied into a @fn closure
        copy x
        //~^ ERROR copying a noncopyable value
    };
    (move x,f)
}
fn closure3(+x: util::NonCopyable) {
    do task::spawn |copy x| {
        //~^ ERROR copying a noncopyable value
        //~^^ NOTE non-copyable value cannot be copied into a ~fn closure
        error!("%?", x);
    }
    error!("%?", x);
}
fn main() {
}
