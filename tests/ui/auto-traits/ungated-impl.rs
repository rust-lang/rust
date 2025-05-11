auto trait MyTrait {}
//~^ ERROR auto traits are experimental and possibly buggy

impl<T> !MyTrait for *mut T {}
//~^ ERROR negative trait bounds are not fully implemented

fn main() {}
