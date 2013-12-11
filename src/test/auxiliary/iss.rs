// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[pkgid="issue6919_3#0.1"];
// NOTE: remove after the next snapshot
#[link(name="iss6919_3", vers="0.1")];

// part of issue-6919.rs

struct C<'self> {
    k: 'self ||,
}

fn no_op() { }
pub static D : C<'static> = C {
    k: no_op
};

