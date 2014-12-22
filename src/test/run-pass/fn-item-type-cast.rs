// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test explicit coercions from a fn item type to a fn pointer type.

fn foo(x: int) -> int { x * 2 }
fn bar(x: int) -> int { x * 4 }
type IntMap = fn(int) -> int;

fn eq<T>(x: T, y: T) { }

static TEST: Option<IntMap> = Some(foo as IntMap);

fn main() {
    let f = foo as IntMap;

    let f = if true { foo as IntMap } else { bar as IntMap };
    assert_eq!(f(4), 8);

    eq(foo as IntMap, bar as IntMap);
}
