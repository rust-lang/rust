#![feature(specialization)]
#![feature(negative_impls)]

trait MyTrait {}

impl<T> !MyTrait for T {}
impl MyTrait for u32 {} //~ ERROR E0748

fn main() {}
