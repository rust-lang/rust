use std::splat::splat; //~ ERROR use of unstable library feature `splat`

#[splat] //~ ERROR use of unstable library feature `splat`
fn tuple_args((a, b, c): (u32, i8, char)) {}

fn main() {}
