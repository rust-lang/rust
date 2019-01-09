use std::fmt::Debug;

trait MultiRegionTrait<'a, 'b> {}
impl<'a, 'b> MultiRegionTrait<'a, 'b> for (&'a u32, &'b u32) {}

fn no_least_region<'a, 'b>(x: &'a u32, y: &'b u32) -> impl MultiRegionTrait<'a, 'b> {
//~^ ERROR ambiguous lifetime bound
    (x, y)
}

fn main() {}
