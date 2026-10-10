type Foo<T,T> = Option<T>;
//~^ ERROR the name `T` is already used

struct Bar<T,T>(T);
//~^ ERROR the name `T` is already used

struct Baz<T,T> {
//~^ ERROR the name `T` is already used
    x: T,
}

enum Boo<T,T> {
//~^ ERROR the name `T` is already used
    A(T),
    B,
}

fn quux<T,T>(x: T) {}
//~^ ERROR the name `T` is already used

trait Qux<T,T> {}
//~^ ERROR the name `T` is already used

impl<T,T> Qux<T,T> for Option<T> {}
//~^ ERROR the name `T` is already used

fn main() {
}
