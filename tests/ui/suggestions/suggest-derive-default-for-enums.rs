// A test showing that we suggest deriving `Default` for enums.
enum MyEnum {
    A,
}

trait Foo {
    fn bar(&self) {}
}
impl<T: Default> Foo for T {}

fn main() {
    let x = MyEnum::A;
    x.bar();
    //~^ ERROR the method `bar` exists for enum `MyEnum`, but its trait bounds were not satisfied
}
