trait MyTrait {}
impl !MyTrait for u32 {} //~ ERROR negative trait bounds are not fully implemented
fn main() {}
