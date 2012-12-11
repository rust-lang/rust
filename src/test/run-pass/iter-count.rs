// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    assert [].count(&22u) == 0u;
    assert [1u, 3u].count(&22u) == 0u;
    assert [22u, 1u, 3u].count(&22u) == 1u;
    assert [22u, 1u, 22u].count(&22u) == 2u;
    assert None.count(&22u) == 0u;
    assert Some(1u).count(&22u) == 0u;
    assert Some(22u).count(&22u) == 1u;
}
