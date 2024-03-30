pub trait Foo<'a> {
    type Assoc;

    fn demo() -> impl Foo
    //~^ ERROR missing lifetime specifier
    //~| ERROR the trait bound `String: Copy` is not satisfied
    where
        String: Copy;
    //~^ ERROR the trait bound `String: Copy` is not satisfied
}

fn main() {}
