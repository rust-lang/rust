trait MyTrait {}

impl !MyTrait for u32 {} //~ ERROR negative impls are experimental

fn main() {}
