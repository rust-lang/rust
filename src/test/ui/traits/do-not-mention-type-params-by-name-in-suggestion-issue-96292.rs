struct Thing<X>(X);

trait Method<T> {
    fn method(self) -> T;
}

impl<X> Method<i32> for Thing<X> {
    fn method(self) -> i32 { 0 }
}

impl<X> Method<u32> for Thing<X> {
    fn method(self) -> u32 { 0 }
}

fn main() {
    let thing = Thing(true);
    thing.method();
    //~^ ERROR type annotations needed
    //~| ERROR type annotations needed
}
