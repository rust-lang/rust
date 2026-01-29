//@ check-fail

pub struct Foo;

pub fn get_foo1<'a>(foo: &'a mut Option<&'a mut Foo>) -> &'a mut Foo {
    match foo {
        &mut Some(ref mut x) => *x,
        //~^ ERROR cannot move out of
        &mut None => panic!(),
    }
}

pub fn get_foo2<'a>(foo: &'a mut Option<&'a mut Foo>) -> &'a mut Foo {
    match foo {
        &mut None => panic!(),
        &mut Some(ref mut x) => *x,
        //~^ ERROR cannot move out of
    }
}

fn main() {
}
