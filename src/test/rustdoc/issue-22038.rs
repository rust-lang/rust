// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern {
    // @has issue_22038/fn.foo1.html \
    //      '//*[@class="rust fn"]' 'pub unsafe extern fn foo1()'
    pub fn foo1();
}

extern "system" {
    // @has issue_22038/fn.foo2.html \
    //      '//*[@class="rust fn"]' 'pub unsafe extern "system" fn foo2()'
    pub fn foo2();
}

// @has issue_22038/fn.bar.html \
//      '//*[@class="rust fn"]' 'pub extern fn bar()'
pub extern fn bar() {}

// @has issue_22038/fn.baz.html \
//      '//*[@class="rust fn"]' 'pub extern "system" fn baz()'
pub extern "system" fn baz() {}
