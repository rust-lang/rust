trait Hello {}

struct Foo<'a, T: ?Sized>(&'a T);

impl<'a, T: ?Sized> Hello for Foo<'a, &'a T> where Foo<'a, T>: Hello {}

impl Hello for Foo<'static, i32> {}

fn hello<T: ?Sized + Hello>() {}

fn main() {
    hello();
    //~^ ERROR type annotations needed
}
