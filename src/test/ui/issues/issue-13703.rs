// build-pass (FIXME(62277): could be check-pass?)
// pretty-expanded FIXME #23616

pub struct Foo<'a, 'b: 'a> { foo: &'a &'b isize }
pub fn foo<'a, 'b>(x: Foo<'a, 'b>, _o: Option<&   &   ()>) { let _y = x.foo; }
fn main() {}
