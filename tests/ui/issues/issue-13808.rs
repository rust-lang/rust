// run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
// pretty-expanded FIXME #23616

struct Foo<'a> {
    listener: Box<dyn FnMut() + 'a>,
}

impl<'a> Foo<'a> {
    fn new<F>(listener: F) -> Foo<'a> where F: FnMut() + 'a {
        Foo { listener: Box::new(listener) }
    }
}

fn main() {
    let a = Foo::new(|| {});
}
