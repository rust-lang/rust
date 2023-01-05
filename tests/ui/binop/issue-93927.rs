// Regression test for #93927: suggested trait bound for T should be Eq, not PartialEq
struct MyType<T>(T);

impl<T> PartialEq for MyType<T>
where
    T: Eq,
{
    fn eq(&self, other: &Self) -> bool {
        true
    }
}

fn cond<T: PartialEq>(val: MyType<T>) -> bool {
    val == val
    //~^ ERROR binary operation `==` cannot be applied to type `MyType<T>`
}

fn main() {
    cond(MyType(0));
}
