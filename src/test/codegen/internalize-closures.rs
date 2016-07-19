// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -C no-prepopulate-passes

pub fn main() {

    // We want to make sure that closures get 'internal' linkage instead of
    // 'weak_odr' when they are not shared between codegen units
    // CHECK: define internal {{.*}}_ZN20internalize_closures4main{{.*}}$u7b$$u7b$closure$u7d$$u7d$
    let c = |x:i32| { x + 1 };
    let _ = c(1);
}
