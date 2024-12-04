trait SendEqAlias<T> = PartialEq;
//~^ ERROR trait aliases are experimental

struct Foo<T>(dyn SendEqAlias<T>);
//~^ ERROR the trait alias `SendEqAlias` cannot be made into an object

struct Bar<T>(dyn SendEqAlias<T>, T);
//~^ ERROR the trait alias `SendEqAlias` cannot be made into an object

fn main() {}
