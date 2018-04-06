// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -C no-prepopulate-passes -Zshare-generics=yes

// Check that local generics are internalized if they are in the same CGU

// CHECK: define internal {{.*}} @_ZN34local_generics_in_exe_internalized3foo{{.*}}
pub fn foo<T>(x: T, y: T) -> (T, T) {
    (x, y)
}

fn main() {
    let _ = foo(0u8, 1u8);
}
