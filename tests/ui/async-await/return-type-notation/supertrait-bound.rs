// check-pass
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![feature(return_position_impl_trait_in_trait, return_type_notation)]
//~^ WARN the feature `return_type_notation` is incomplete and may not be safe to use

trait IntFactory {
    fn stream(&self) -> impl Iterator<Item = i32>;
}
trait SendIntFactory: IntFactory<stream(): Send> + Send {}

fn main() {}
