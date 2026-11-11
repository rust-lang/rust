// Regression test for https://github.com/rust-lang/rust/issues/81447
// Checks that the lint for private fields is prioritized over the self lint.

mod some_module {
    pub struct Test<T: ?Sized>(T);
}

use some_module::Test;

struct TestBuilder;

impl TestBuilder {
    fn build(self) -> Test {
                      //~^ ERROR missing generics for struct `Test`
        Test(self)
        //~^ ERROR cannot initialize a tuple struct which contains private fields
    }
}

fn main() {}
