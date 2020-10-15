struct Foo<T, U = [u8; std::mem::size_of::<T>()]>(T, U);
//~^ ERROR constant values inside of type parameter defaults

fn main() {}
