// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

struct Foo { foo: bool, bar: Option<isize>, baz: isize }

pub fn main() {
    match (Foo{foo: true, bar: Some(10), baz: 20}) {
      Foo{foo: true, bar: Some(_), ..} => {}
      Foo{foo: false, bar: None, ..} => {}
      Foo{foo: true, bar: None, ..} => {}
      Foo{foo: false, bar: Some(_), ..} => {}
    }
}
