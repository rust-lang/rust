// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn of<'a,T>() -> |T|:'a { panic!(); }
fn subtype<T>(x: |T|) { panic!(); }

fn test_fn<'x,'y,'z,T>(_x: &'x T, _y: &'y T, _z: &'z T) {
    // Here, x, y, and z are free.  Other letters
    // are bound.  Note that the arrangement
    // subtype::<T1>(of::<T2>()) will typecheck
    // iff T1 <: T2.

    subtype::< for<'a>|&'a T|>(
        of::< for<'a>|&'a T|>());

    subtype::< for<'a>|&'a T|>(
        of::< for<'b>|&'b T|>());

    subtype::< for<'b>|&'b T|>(
        of::<|&'x T|>());

    subtype::<|&'x T|>(
        of::< for<'b>|&'b T|>());  //~ ERROR mismatched types

    subtype::< for<'a,'b>|&'a T, &'b T|>(
        of::< for<'a>|&'a T, &'a T|>());

    subtype::< for<'a>|&'a T, &'a T|>(
        of::< for<'a,'b>|&'a T, &'b T|>()); //~ ERROR mismatched types

    subtype::< for<'a,'b>|&'a T, &'b T|>(
        of::<|&'x T, &'y T|>());

    subtype::<|&'x T, &'y T|>(
        of::< for<'a,'b>|&'a T, &'b T|>()); //~ ERROR mismatched types
}

fn main() {}
