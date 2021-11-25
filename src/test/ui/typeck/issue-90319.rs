struct Wrapper<T>(T);

trait Trait {
    fn method(&self) {}
}

impl<'a, T> Trait for Wrapper<&'a T> where Wrapper<T>: Trait {}

fn get<T>() -> T {
    unimplemented!()
}

fn main() {
    let thing = get::<Thing>();//~ERROR 14:23: 14:28: cannot find type `Thing` in this scope [E0412]
    let wrapper = Wrapper(thing);
    Trait::method(&wrapper);//~ERROR 16:5: 16:18: overflow evaluating the requirement `_: Sized` [E0275]
}
