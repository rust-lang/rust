// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test associated types appearing in struct-like enum variants.

// pretty-expanded FIXME #23616

use self::VarValue::*;

pub trait UnifyKey {
    type Value;
    fn to_index(&self) -> usize;
}

pub enum VarValue<K:UnifyKey> {
    Redirect { to: K },
    Root { value: K::Value, rank: usize },
}

fn get<'a,K:UnifyKey<Value=Option<V>>,V>(table: &'a Vec<VarValue<K>>, key: &K) -> &'a Option<V> {
    match table[key.to_index()] {
        VarValue::Redirect { to: ref k } => get(table, k),
        VarValue::Root { value: ref v, rank: _ } => v,
    }
}

impl UnifyKey for usize {
    type Value = Option<char>;
    fn to_index(&self) -> usize { *self }
}

fn main() {
    let table = vec![/* 0 */ Redirect { to: 1 },
                     /* 1 */ Redirect { to: 3 },
                     /* 2 */ Root { value: Some('x'), rank: 0 },
                     /* 3 */ Redirect { to: 2 }];
    assert_eq!(get(&table, &0), &Some('x'));
}
