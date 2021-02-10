mod some_module {
    pub struct Test<T: ?Sized>(T);
}

use some_module::Test;

struct TestBuilder<T> {
    something: T,
}

impl<T> TestBuilder<T> {
    fn build(self) -> Test<T> {
        Test(self)
        //~^ ERROR cannot initialize a tuple struct which contains private fields
    }
}

fn main() {}
