// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::marker::PhantomData;

pub trait DataBind {
    type Data;
}

impl<T> DataBind for Global<T> {
    type Data = T;
}

pub struct Global<T>(PhantomData<T>);

pub struct Data {
    pub offsets: <Global<[u32; 2]> as DataBind>::Data,
}

fn create_data() -> Data {
    let mut d = Data { offsets: [1, 2] };
    d.offsets[0] = 3;
    d
}


fn main() {
    let d = create_data();
    assert_eq!(d.offsets[0], 3);
    assert_eq!(d.offsets[1], 2);
}
