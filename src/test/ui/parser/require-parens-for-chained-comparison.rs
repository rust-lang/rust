fn main() {
    false == false == false;
    //~^ ERROR comparison operators cannot be chained
    //~| HELP split the comparison into two

    false == 0 < 2;
    //~^ ERROR comparison operators cannot be chained
    //~| HELP parenthesize the comparison

    f<X>();
    //~^ ERROR comparison operators cannot be chained
    //~| HELP use `::<...>` instead of `<...>` to specify lifetime, type, or const arguments

    f<Result<Option<X>, Option<Option<X>>>(1, 2);
    //~^ ERROR comparison operators cannot be chained
    //~| HELP use `::<...>` instead of `<...>` to specify lifetime, type, or const arguments

    let _ = f<u8, i8>();
    //~^ ERROR expected one of
    //~| HELP use `::<...>` instead of `<...>` to specify lifetime, type, or const arguments

    let _ = f<'_, i8>();
    //~^ ERROR expected one of
    //~| HELP use `::<...>` instead of `<...>` to specify lifetime, type, or const arguments
    //~| ERROR expected

    f<'_>();
    //~^ comparison operators cannot be chained
    //~| HELP use `::<...>` instead of `<...>` to specify lifetime, type, or const arguments
    //~| ERROR expected

    let _ = f<u8>;
    //~^ ERROR comparison operators cannot be chained
    //~| HELP use `::<...>` instead of `<...>` to specify lifetime, type, or const arguments
    //~| HELP or use `(...)` if you meant to specify fn arguments
}
