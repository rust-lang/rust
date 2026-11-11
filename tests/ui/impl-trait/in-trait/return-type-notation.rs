//@ check-pass

#![allow(incomplete_features)]
#![feature(return_type_notation)]

trait IntFactory {
    fn stream(&self) -> impl IntFactory<stream(..): IntFactory<stream(..): Send> + Send>;
}

pub fn main() {}
