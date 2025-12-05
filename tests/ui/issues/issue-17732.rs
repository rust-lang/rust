//@ check-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]

trait Person {
    type string;
    fn dummy(&self) { }
}

struct Someone<P: Person>(std::marker::PhantomData<P>);

fn main() {}
