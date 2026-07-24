pub struct Foo<'a, T>(&'a T);
fn foo<'a, 'b: 'a, T: 'b + 'a>(this: &'a Foo<'b, T>) {}

fn main() {}