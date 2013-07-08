// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(default_methods)]
trait Speak {
    fn say(&self, s:&str) -> ~str;
    fn hi(&self) -> ~str { hello(self) }
}

fn hello<S:Speak>(s:&S) -> ~str{
    s.say("hello")
}

impl Speak for int {
    fn say(&self, s:&str) -> ~str {
        fmt!("%s: %d", s, *self)
    }
}

impl<T: Speak> Speak for Option<T> {
    fn say(&self, s:&str) -> ~str {
        match *self {
            None => fmt!("%s - none", s),
            Some(ref x) => { ~"something!" + x.say(s) }
        }
    }
}


fn main() {
    assert_eq!(3.hi(), ~"hello: 3");
    assert_eq!(Some(Some(3)).hi(), ~"something!something!hello: 3");
    assert_eq!(None::<int>.hi(), ~"hello - none");

    assert_eq!(Some(None::<int>).hi(), ~"something!hello - none");
    assert_eq!(Some(3).hi(), ~"something!hello: 3");
}
