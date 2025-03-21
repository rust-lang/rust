#![deny(unused_must_use)]

fn any<T>() -> T {
    todo!()
}
fn it() -> impl ExactSizeIterator<Item = ()> {
    let x: Box<dyn ExactSizeIterator<Item = ()>> = any();
    x
}

fn main() {
    it(); //~ ERROR unused implementer of `Iterator` that must be used
}
