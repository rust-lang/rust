//@ check-pass
#![allow(dead_code)]
#![allow(type_alias_bounds)]

type Foo<T> where T: Copy = Box<T>;

fn main(){}
