// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let value = Some(1i);
    assert_eq!(match value {
        ref a @ Some(_) => a,
        ref b @ None => b
    }, &Some(1i));
    assert_eq!(match value {
        ref c @ Some(_) => c,
        ref b @ None => b
    }, &Some(1i));
    assert_eq!(match "foobarbaz" {
        b @ _ => b
    }, "foobarbaz");
    let a @ _ = "foobarbaz";
    assert_eq!(a, "foobarbaz");
    let value = Some(true);
    let ref a @ _ = value;
    assert_eq!(a, &Some(true));
}
