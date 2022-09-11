trait SendEqAlias<T> = PartialEq;
//~^ ERROR trait aliases are experimental

struct Foo<T>(dyn SendEqAlias<T>);
//~^ ERROR the type parameter `Rhs` must be explicitly specified [E0393]

struct Bar<T>(dyn SendEqAlias<T>, T);
//~^ ERROR the type parameter `Rhs` must be explicitly specified [E0393]

fn main() {}
