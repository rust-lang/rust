trait Foo: Iterator<Item = i32, Item = i32> {}
//~^ ERROR is already specified

type Unit = ();

fn test() -> Box<Iterator<Item = (), Item = Unit>> {
//~^ ERROR is already specified
    Box::new(None.into_iter())
}

fn main() {
    let _: &Iterator<Item = i32, Item = i32>;
    test();
}
