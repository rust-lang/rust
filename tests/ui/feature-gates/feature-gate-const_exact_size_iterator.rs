//@ check-fail

#![feature(const_trait_impl)]
#![feature(const_iter)]


fn main() {}


struct MyType;

const impl Iterator for MyType {
    type Item = ();
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

const impl ExactSizeIterator for MyType {}
