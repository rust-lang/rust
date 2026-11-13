trait Foo {
    fn t1() -> [u8; std::mem::size_of::<Self>()]; //~ERROR generic parameters
}

struct Bar<T>(T);

impl Bar<u8> {
    fn t2() -> [u8; std::mem::size_of::<Self>()] { todo!() } // ok
}

impl<T> Bar<T> {
    fn t3() -> [u8; std::mem::size_of::<Self>()] {} //~ERROR generic `Self`
}

trait Baz {
    fn hey();
}

impl Baz for u16 {
    fn hey() {
        let _: [u8; std::mem::size_of::<Self>()]; // ok
    }
}

fn main() {}
