// Test that you cannot *directly* dispatch on lifetime requirements

trait MyTrait { fn foo() {} }

impl<T> MyTrait for T {}
impl<T: 'static> MyTrait for T {}
//~^ ERROR E0119

fn main() {}
