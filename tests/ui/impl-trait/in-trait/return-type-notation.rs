#![allow(incomplete_features)]
trait IntFactory {
    fn stream(&self) -> impl IntFactory<stream(..): IntFactory<stream(..): Send> + Send>;
    //~^ ERROR cycle detected when resolving lifetimes for `IntFactory::stream`
}

pub fn main() {}
