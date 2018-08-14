// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that we check fns appearing in constant declarations.
// Issue #22382.

// Returning local references?
struct DropString {
    inner: String
}
impl Drop for DropString {
    fn drop(&mut self) {
        self.inner.clear();
        self.inner.push_str("dropped");
    }
}
const LOCAL_REF: fn() -> &'static str = {
    fn broken() -> &'static str {
        let local = DropString { inner: format!("Some local string") };
        return &local.inner; //~ ERROR does not live long enough
    }
    broken
};

fn main() {
}
