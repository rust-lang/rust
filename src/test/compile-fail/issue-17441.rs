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
    let _foo = &[1u, 2] as [uint];
    //~^ ERROR cast to unsized type: `&[uint, .. 2]` as `[uint]`
    //~^^ NOTE consider using an implicit coercion to `&[uint]` instead
    let _bar = box 1u as std::fmt::Show;
    //~^ ERROR cast to unsized type: `Box<uint>` as `core::fmt::Show`
    //~^^ NOTE did you mean `Box<core::fmt::Show>`?
    let _baz = 1u as std::fmt::Show;
    //~^ ERROR cast to unsized type: `uint` as `core::fmt::Show`
    //~^^ NOTE consider using a box or reference as appropriate
    let _quux = [1u, 2] as [uint];
    //~^ ERROR cast to unsized type: `[uint, .. 2]` as `[uint]`
    //~^^ NOTE consider using a box or reference as appropriate
}
