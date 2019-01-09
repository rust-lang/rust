fn f<T>() {}
struct X;

fn main() {
    false == false == false;
    //~^ ERROR: chained comparison operators require parentheses

    false == 0 < 2;
    //~^ ERROR: chained comparison operators require parentheses
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types

    f<X>();
    //~^ ERROR: chained comparison operators require parentheses
    //~| ERROR: binary operation `<` cannot be applied to type `fn() {f::<_>}`
    //~| HELP: use `::<...>` instead of `<...>`
    //~| HELP: or use `(...)`
}
