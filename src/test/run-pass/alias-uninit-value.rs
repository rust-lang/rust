// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.



// Regression test for issue #374

enum sty { ty_nil, }

struct RawT {struct_: sty, cname: Option<StrBuf>, hash: uint}

fn mk_raw_ty(st: sty, cname: Option<StrBuf>) -> RawT {
    return RawT {struct_: st, cname: cname, hash: 0u};
}

pub fn main() { mk_raw_ty(ty_nil, None::<StrBuf>); }
