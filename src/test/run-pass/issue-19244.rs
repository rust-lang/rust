// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

struct MyStruct { field: usize }
struct Nested { nested: MyStruct }
struct Mix2 { nested: ((usize,),) }

const STRUCT: MyStruct = MyStruct { field: 42 };
const TUP: (usize,) = (43,);
const NESTED_S: Nested = Nested { nested: MyStruct { field: 5 } };
const NESTED_T: ((usize,),) = ((4,),);
const MIX_1: ((Nested,),) = ((Nested { nested: MyStruct { field: 3 } },),);
const MIX_2: Mix2 = Mix2 { nested: ((2,),) };
const INSTANT_1: usize = (MyStruct { field: 1 }).field;
const INSTANT_2: usize = (0,).0;

fn main() {
    let a = [0; STRUCT.field];
    let b = [0; TUP.0];
    let c = [0; NESTED_S.nested.field];
    let d = [0; (NESTED_T.0).0];
    let e = [0; (MIX_1.0).0.nested.field];
    let f = [0; (MIX_2.nested.0).0];
    let g = [0; INSTANT_1];
    let h = [0; INSTANT_2];

    assert_eq!(a.len(), 42);
    assert_eq!(b.len(), 43);
    assert_eq!(c.len(), 5);
    assert_eq!(d.len(), 4);
    assert_eq!(e.len(), 3);
    assert_eq!(f.len(), 2);
    assert_eq!(g.len(), 1);
    assert_eq!(h.len(), 0);
}
