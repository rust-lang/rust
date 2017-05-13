// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z parse-only -Z continue-parse-after-error

struct Baz<U> where U: Eq(U); //This is parsed as the new Fn* style parenthesis syntax.
struct Baz<U> where U: Eq(U) -> R; // Notice this parses as well.
struct Baz<U>(U) where U: Eq; // This rightfully signals no error as well.
struct Foo<T> where T: Copy, (T); //~ ERROR unexpected token in `where` clause
struct Bar<T> { x: T } where T: Copy //~ ERROR expected item, found `where`

fn main() {}
