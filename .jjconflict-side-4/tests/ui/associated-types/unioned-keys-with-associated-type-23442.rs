// https://github.com/rust-lang/rust/issues/23442
//@ check-pass
#![allow(dead_code)]
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
