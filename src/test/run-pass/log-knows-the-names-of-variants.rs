// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum foo {
  a(uint),
  b(StrBuf),
  c,
}

enum bar {
  d, e, f
}

pub fn main() {
    assert_eq!("a(22u)".to_strbuf(), format_strbuf!("{:?}", a(22u)));
    assert_eq!("c".to_strbuf(), format_strbuf!("{:?}", c));
    assert_eq!("d".to_strbuf(), format_strbuf!("{:?}", d));
}
