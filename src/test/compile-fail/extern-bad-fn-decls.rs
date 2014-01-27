// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Concerning issue #10877

struct Foo { x: int }

extern {

    fn external_fn_one(1: ()); //~ ERROR expected ident, found `1`

    fn external_fn_two((): int); //~ ERROR expected ident, found `(`

    fn external_fn_three(Foo {x}: int); //~ ERROR expected `:`, found `{`

    fn external_fn_four((x,y): int); //~ ERROR expected ident, found `(`

    fn external_fn_five(_: int); //~ ERROR expected ident, found `_`

    fn external_fn_six(~int); //~ ERROR expected ident, found `~`
}

fn main() {}
