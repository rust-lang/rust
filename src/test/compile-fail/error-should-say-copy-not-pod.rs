// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that the error message uses the word Copy, not Pod.

fn check_bound<T:Copy>(_: T) {}

fn main() {
    check_bound("nocopy".to_string()); //~ ERROR the trait `core::marker::Copy` is not implemented
}
