#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

// check-pass

use std::fmt::Debug;

type Foo = impl Debug;

#[defines(Foo)]
static FOO1: Foo = 22_u32;
#[defines(Foo)]
const FOO2: Foo = 22_u32;

fn main() {}
