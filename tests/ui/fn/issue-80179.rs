// Functions with a type placeholder `_` as the return type should
// show a function pointer suggestion when given a function item
// and suggest how to return closures correctly from a function.
// This is a regression test of #80179

fn returns_i32() -> i32 {
    0
}

fn returns_fn_ptr() -> _ {
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types [E0121]
//~| NOTE not allowed in type signatures
//~| HELP replace with the correct return type
//~| SUGGESTION fn() -> i32
    returns_i32
}

fn returns_closure() -> _ {
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types [E0121]
//~| NOTE not allowed in type signatures
//~| HELP replace with an appropriate return type
//~| SUGGESTION impl Fn() -> i32
//~| NOTE for more information on `Fn` traits and closure types
    || 0
}

fn main() {}
