struct Foo<T>(T, *const ());

unsafe impl Send for Foo<T> {}
//~^ ERROR cannot find type

fn main() {}
