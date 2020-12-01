fn f<T>() {}
struct X;

fn main() {
    false == false == false;
    //~^ ERROR comparison operators cannot be chained
    //~| HELP split the comparison into two

    false == 0 < 2;
    //~^ ERROR comparison operators cannot be chained
    //~| HELP parenthesize the comparison

    f<X>();
    //~^ ERROR comparison operators cannot be chained
    //~| HELP use `::<...>` instead of `<...>` to specify type or const arguments

    f<Result<Option<X>, Option<Option<X>>>(1, 2);
    //~^ ERROR comparison operators cannot be chained
    //~| HELP use `::<...>` instead of `<...>` to specify type or const arguments

    use std::convert::identity;
    let _ = identity<u8>;
    //~^ ERROR comparison operators cannot be chained
    //~| HELP use `::<...>` instead of `<...>` to specify type or const arguments
    //~| HELP or use `(...)` if you meant to specify fn arguments
}
