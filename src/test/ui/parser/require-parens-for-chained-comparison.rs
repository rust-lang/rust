// compile-flags: -Z parse-only

fn f<T>() {}

fn main() {
    false == false == false;
    //~^ ERROR: chained comparison operators require parentheses

    false == 0 < 2;
    //~^ ERROR: chained comparison operators require parentheses

    f<X>();
    //~^ ERROR: chained comparison operators require parentheses
    //~| HELP: use `::<...>` instead of `<...>`
    //~| HELP: or use `(...)`
}
