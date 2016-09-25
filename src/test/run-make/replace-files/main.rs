// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test is run with -Zreplace-files and we pipe a replacement for
// "/foo/foo.rs" to stdin. The compiler should read that and use the replacement
// text when it sees the `path` attribute below.
//
// If we actually try and look up /foo/foo.rs (which presumably will cause a
// missing file error), then the replace-files code is broken.

#[path="/foo/foo.rs"]
mod foo;

fn main() {
    assert!(foo::foo());
}
