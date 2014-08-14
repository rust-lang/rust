// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Various tests where we over type parameters with multiple lifetime
// bounds.

trait SomeTrait { fn get(&self) -> int; }

fn make_object_good1<'a,'b,A:SomeTrait+'a+'b>(v: A) -> Box<SomeTrait+'a> {
    // A outlives 'a AND 'b...
    box v as Box<SomeTrait+'a> // ...hence this type is safe.
}

fn make_object_good2<'a,'b,A:SomeTrait+'a+'b>(v: A) -> Box<SomeTrait+'b> {
    // A outlives 'a AND 'b...
    box v as Box<SomeTrait+'b> // ...hence this type is safe.
}

fn make_object_bad<'a,'b,'c,A:SomeTrait+'a+'b>(v: A) -> Box<SomeTrait+'c> {
    // A outlives 'a AND 'b...but not 'c.
    box v as Box<SomeTrait+'a> //~ ERROR mismatched types
}

fn main() {
}
