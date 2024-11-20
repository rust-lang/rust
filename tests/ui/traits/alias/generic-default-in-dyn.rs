trait SendEqAlias<T> = PartialEq;
//~^ ERROR trait aliases are experimental

struct Foo<T>(dyn SendEqAlias<T>);
//~^ ERROR the trait alias `SendEqAlias` is not dyn compatible

struct Bar<T>(dyn SendEqAlias<T>, T);
//~^ ERROR the trait alias `SendEqAlias` is not dyn compatible

fn main() {}
