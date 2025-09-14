// https://github.com/rust-lang/rust/issues/56870
//@ build-pass
// Regression test for #56870: Internal compiler error (traits & associated consts)

use std::fmt::Debug;

pub trait Foo<T> {
  const FOO: *const u8;
}

impl <T: Debug> Foo<T> for dyn Debug {
  const FOO: *const u8 = <T as Debug>::fmt as *const u8;
}

pub trait Bar {
  const BAR: *const u8;
}

pub trait Baz {
  type Data: Debug;
}

pub struct BarStruct<S: Baz>(S);

impl<S: Baz> Bar for BarStruct<S> {
  const BAR: *const u8 = <dyn Debug as Foo<<S as Baz>::Data>>::FOO;
}

struct AnotherStruct;
#[derive(Debug)]
struct SomeStruct;

impl Baz for AnotherStruct {
  type Data = SomeStruct;
}

fn main() {
  let _x = <BarStruct<AnotherStruct> as Bar>::BAR;
}
