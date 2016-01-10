// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z parse-only

struct Foo<'a, 'b> {
    a: &'a &'b i32
}

fn foo<'a, 'b>(x: &mut Foo<'a; 'b>) {}
//~^ ERROR expected `,` or `>` after lifetime name, found `;`
//~^^ NOTE did you mean a single argument type &'a Type, or did you mean the comma-separated
