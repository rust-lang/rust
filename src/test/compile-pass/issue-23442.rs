// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::marker::PhantomData;

pub struct UnionedKeys<'a,K>
    where K: UnifyKey + 'a
{
    table: &'a mut UnificationTable<K>,
    root_key: K,
    stack: Vec<K>,
}

pub trait UnifyKey {
    type Value;
}

pub struct UnificationTable<K:UnifyKey> {
    values: Delegate<K>,
}

pub struct Delegate<K>(PhantomData<K>);

fn main() {}
