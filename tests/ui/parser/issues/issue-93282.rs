fn main() {
    //~^ HELP consider introducing lifetime `'a` here
    f<'a,>
    //~^ ERROR use `::<...>` instead of `<...>` to specify lifetime arguments
    //~| HELP add `::`
    //~| ERROR use of undeclared lifetime name `'a`
    //~| ERROR cannot find value `f` in this scope
}

fn bar(a: usize, b: usize) -> usize {
    a + b
}

fn foo() {
    let x = 1;
    bar('y, x);
    //~^ ERROR expected
    //~| HELP add `'` to close the char literal
    //~| ERROR mismatched types
}
