// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

static mut DROP: bool = false;

struct ConnWrap(Conn);
impl ::std::ops::Deref for ConnWrap {
    type Target=Conn;
    fn deref(&self) -> &Conn { &self.0 }
}

struct Conn;
impl Drop for  Conn {
    fn drop(&mut self) { unsafe { DROP = true; } }
}

fn inner() {
    let conn = &*match Some(ConnWrap(Conn)) {
        Some(val) => val,
        None => return,
    };
    return;
}

fn main() {
    inner();
    unsafe {
        assert_eq!(DROP, true);
    }
}
