// check-pass
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![feature(return_position_impl_trait_in_trait)]

trait Foo {}

impl Foo for () {}

trait ThreeCellFragment {
    fn ext_cells<'a>(&'a self) -> impl Foo + 'a {
        self.ext_adjacent_cells()
    }

    fn ext_adjacent_cells<'a>(&'a self) -> impl Foo + 'a;
}

fn main() {}
