//@ compile-flags: -Znext-solver

#![feature(const_trait_impl)]

#[const_trait]
trait MyTrait {
    fn do_something(&self);
}

trait OtherTrait {
    fn do_something_else() where Self: [const] MyTrait;
    //~^ ERROR `[const]` is not allowed here
}

struct MyStruct<T>(T);

impl const MyTrait for u32 {
    fn do_something(&self) {}
}

impl<T> MyStruct<T> {
    pub fn foo(&self) where T: [const] MyTrait {
        //~^ ERROR `[const]` is not allowed here
        self.0.do_something();
    }
}

fn main() {
    MyStruct(0u32).foo();
}
