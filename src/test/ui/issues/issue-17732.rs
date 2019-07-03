// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
// pretty-expanded FIXME #23616

trait Person {
    type string;
    fn dummy(&self) { }
}

struct Someone<P: Person>(std::marker::PhantomData<P>);

fn main() {}
