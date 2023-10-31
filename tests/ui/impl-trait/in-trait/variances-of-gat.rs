// check-pass

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
