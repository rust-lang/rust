//@ run-pass
#![allow(dead_code)]
#![allow(unused_variables)]

trait U {}
trait T<X: U> { fn get(self) -> X; }

trait S2<Y: U> {
    fn m(x: Box<dyn T<Y>+'static>) {}
}

struct St<X: U> {
    f: Box<dyn T<X>+'static>,
}

impl<X: U> St<X> {
    fn blah() {}
}

fn main() {}
