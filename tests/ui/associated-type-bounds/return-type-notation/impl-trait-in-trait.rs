#![feature(return_type_notation)]

trait IntFactory {
    fn stream(self) -> impl IntFactory<stream(..): Send>;
    //~^ ERROR cycle detected when resolving lifetimes for `IntFactory::stream`
}

fn main() {}
