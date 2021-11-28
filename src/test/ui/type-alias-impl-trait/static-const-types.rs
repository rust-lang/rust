#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

// FIXME: This should compile, but it currently doesn't

use std::fmt::Debug;

type Foo = impl Debug; //~ ERROR could not find defining uses

static FOO1: Foo = 22_u32; //~ ERROR mismatched types
const FOO2: Foo = 22_u32; //~ ERROR mismatched types

fn main() {}
