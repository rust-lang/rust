fn f<T>() {}
struct X;

fn main() {
    false == false == false;
    //~^ ERROR chained comparison operators require parentheses

    false == 0 < 2;
    //~^ ERROR chained comparison operators require parentheses
    //~| ERROR mismatched types
    //~| ERROR mismatched types

    f<X>();
    //~^ ERROR chained comparison operators require parentheses
    //~| HELP use `::<...>` instead of `<...>` to specify type arguments

    f<Result<Option<X>, Option<Option<X>>>(1, 2);
    //~^ ERROR chained comparison operators require parentheses
    //~| HELP use `::<...>` instead of `<...>` to specify type arguments

    use std::convert::identity;
    let _ = identity<u8>;
    //~^ ERROR chained comparison operators require parentheses
    //~| HELP use `::<...>` instead of `<...>` to specify type arguments
    //~| HELP or use `(...)` if you meant to specify fn arguments
}
