//@ run-pass

pub struct Foo;

pub fn get_foo2<'a>(foo: &'a mut Option<&'a mut Foo>) -> &'a mut Foo {
    match foo {
        // Ensure that this is not considered a move, but rather a reborrow.
        &mut Some(ref mut x) => *x,
        &mut None => panic!(),
    }
}

fn main() {
}
