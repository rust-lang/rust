struct Foo<T, U = [u8; std::mem::size_of::<T>()]>(T, U);
//~^ ERROR generic parameters may not be used in const operations

fn main() {}
