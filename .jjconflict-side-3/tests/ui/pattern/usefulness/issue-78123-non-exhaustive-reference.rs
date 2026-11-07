enum A {}
    //~^ NOTE `A` defined here

fn f(a: &A) {
    match a {}
    //~^ ERROR non-exhaustive patterns: type `&A` is non-empty
    //~| NOTE the matched value is of type `&A`
    //~| NOTE references are always considered inhabited
}

fn main() {}
