// -*- rust -*-
// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Rec {
    f: int
}

fn f(p: *Rec) -> int {

    // Test that * ptrs do not autoderef.  There is a deeper reason for
    // prohibiting this, beyond making unsafe things annoying (which doesn't
    // actually seem desirable to me).  The deeper reason is that if you
    // have a type like:
    //
    //    enum foo = *foo;
    //
    // you end up with an infinite auto-deref chain, which is
    // currently impossible (in all other cases, infinite auto-derefs
    // are prohibited by various checks, such as that the enum is
    // instantiable and so forth).

    return p.f; //~ ERROR attempted access of field `f` on type `*Rec`
}

fn main() {
}
