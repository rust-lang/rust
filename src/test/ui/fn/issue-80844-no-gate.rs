// Functions with a type placeholder `_` as the return type should
// not suggest anything if generators aren't enabled.

fn returns_generator() -> _ {
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures [E0121]
//~| NOTE not allowed in type signatures
    || yield 0
    //~^ ERROR yield syntax is experimental
    //~| NOTE see issue
}

fn main() {}
