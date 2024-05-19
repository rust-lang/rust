#![feature(specialization)] //~ WARN the feature `specialization` is incomplete
#![feature(negative_impls)]

trait MyTrait {}

impl<T> !MyTrait for T {}
impl MyTrait for u32 {} //~ ERROR E0751

fn main() {}
