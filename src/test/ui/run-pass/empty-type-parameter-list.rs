// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that empty type parameter list (<>) is synonymous with
// no type parameters at all

struct S<>;
trait T<> {}
enum E<> { V }
impl<> T<> for S<> {}
impl T for E {}
fn foo<>() {}
fn bar() {}

fn main() {
    let _ = S;
    let _ = S::<>;
    let _ = E::V;
    let _ = E::<>::V;
    foo();
    foo::<>();

    // Test that we can supply <> to non generic things
    bar::<>();
    let _: i32<>;
}
