trait MyTrait {}
impl !MyTrait for u32 {} //~ ERROR negative trait bounds are not yet fully implemented
fn main() {}
