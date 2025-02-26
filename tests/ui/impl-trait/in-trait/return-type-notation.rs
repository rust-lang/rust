#![allow(incomplete_features)]
#![feature(return_type_notation)]

trait IntFactory {
    fn stream(&self) -> impl IntFactory<stream(..): IntFactory<stream(..): Send> + Send>;
    //~^ ERROR cycle detected when resolving lifetimes for `IntFactory::stream`
}

pub fn main() {}
