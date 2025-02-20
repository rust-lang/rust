//@ check-pass
// ICE: #10972
// asked to assemble constituent types of unexpected type: Binder(Foo, [])
#![feature(type_alias_impl_trait)]

use std::fmt::Debug;
type Foo = impl Debug;
const FOO2: Foo = 22_u32;

pub fn main() {}
