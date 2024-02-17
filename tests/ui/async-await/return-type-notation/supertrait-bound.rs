//@ check-pass

#![feature(return_type_notation)]
//~^ WARN the feature `return_type_notation` is incomplete and may not be safe to use

trait IntFactory {
    fn stream(&self) -> impl Iterator<Item = i32>;
}
trait SendIntFactory: IntFactory<stream(): Send> + Send {}

fn main() {}
