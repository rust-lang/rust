// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that `&PrinterSupport`, which is really short for `&'a
// PrinterSupport<'b>`, gets properly expanded when it appears in a
// closure type. This used to result in messed up De Bruijn indices.

trait PrinterSupport<'ast> {
    fn ast_map(&self) -> Option<&'ast uint> { None }
}

struct NoAnn<'ast> {
    f: Option<&'ast uint>
}

impl<'ast> PrinterSupport<'ast> for NoAnn<'ast> {
}

fn foo<'ast> (f: Option<&'ast uint>, g: |&PrinterSupport|) {
    let annotation = NoAnn { f: f };
    g(&annotation)
}

fn main() {}
