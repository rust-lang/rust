trait Foo: Iterator<Item = i32, Item = i32> {}
//~^ ERROR is already specified

type Unit = ();

fn test() -> Box<dyn Iterator<Item = (), Item = Unit>> {
//~^ ERROR is already specified
    Box::new(None.into_iter())
}

fn main() {
    let _: &dyn Iterator<Item = i32, Item = i32>;
    test();
}
