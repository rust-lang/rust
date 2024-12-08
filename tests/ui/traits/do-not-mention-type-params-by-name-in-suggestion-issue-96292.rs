struct Thing<X>(X);

trait Method<T> {
    fn method(self, _: i32) -> T;
}

impl<X> Method<i32> for Thing<X> {
    fn method(self, _: i32) -> i32 { 0 }
}

impl<X> Method<u32> for Thing<X> {
    fn method(self, _: i32) -> u32 { 0 }
}

fn main() {
    let thing = Thing(true);
    thing.method(42); //~ ERROR type annotations needed
}
