// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test checks that the error messages you get for this example
// at least mention `'a` and `'static`. The precise messages can drift
// over time, but this test used to exhibit some pretty bogus messages
// that were not remotely helpful.

// error-pattern:cannot infer
// error-pattern:cannot outlive the lifetime 'a
// error-pattern:must be valid for the static lifetime
// error-pattern:cannot infer
// error-pattern:cannot outlive the lifetime 'a
// error-pattern:must be valid for the static lifetime

struct Invariant<'a>(Option<&'a mut &'a mut ()>);

fn mk_static() -> Invariant<'static> { Invariant(None) }

fn unify<'a>(x: Option<Invariant<'a>>, f: fn(Invariant<'a>)) {
    let bad = if x.is_some() {
        x.unwrap()
    } else {
        mk_static()
    };
    f(bad);
}

fn main() {}
