#![feature(generators)]

// Functions with a type placeholder `_` as the return type should
// not suggest returning the unnameable type of generators.
// This is a regression test of #80844

struct Container<T>(T);

fn returns_generator() -> _ {
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures [E0121]
//~| NOTE not allowed in type signatures
//~| HELP consider using a `Generator` trait bound
//~| NOTE for more information on generators
    || yield 0i32
}

fn returns_returns_generator() -> _ {
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures [E0121]
//~| NOTE not allowed in type signatures
    returns_generator
}

fn returns_option_closure() -> _ {
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures [E0121]
//~| NOTE not allowed in type signatures
    Some(|| 0i32)
}

fn returns_option_i32() -> _ {
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures [E0121]
//~| NOTE not allowed in type signatures
//~| HELP replace with the correct return type
//~| SUGGESTION Option<i32>
    Some(0i32)
}

fn returns_container_closure() -> _ {
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures [E0121]
//~| NOTE not allowed in type signatures
    Container(|| 0i32)
}

fn returns_container_i32() -> _ {
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures [E0121]
//~| NOTE not allowed in type signatures
//~| HELP replace with the correct return type
//~| SUGGESTION Container<i32>
    Container(0i32)
}

fn main() {}
