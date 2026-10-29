struct Test(i32, i64);
trait Foo<'a> {
    type Assoc;
}

trait SimpleFoo {
    type SimpleAssoc;
}
impl<'a, T> Foo<'a> for T where T: SimpleFoo {
    type Assoc = T::SimpleAssoc;
}

impl SimpleFoo for i32 {
    type SimpleAssoc = i32;
}
impl SimpleFoo for i64 {
    type SimpleAssoc = i32;
}

impl<'a> Foo<'a> for Test where i32: Foo<'a, Assoc = i32>, i64: Foo<'a, Assoc = i64> {
    type Assoc = Test;
}

fn process<'a, T: Foo<'a>>(_input: T) {}
fn test() { process(Test(0, 1)) }
//~^ ERROR the trait bound `Test: Foo<'_>` is not satisfied

fn main() {}
