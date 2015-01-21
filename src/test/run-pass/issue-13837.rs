// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct TestStruct {
    x: *const [int; 2]
}

unsafe impl Sync for TestStruct {}

static TEST_VALUE : TestStruct = TestStruct{x: 0x1234 as *const [int; 2]};

fn main() {}
