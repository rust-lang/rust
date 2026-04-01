// Regression test for #85794
//@ edition: 2015

struct Baz {
    inner : dyn fn ()
    //~^ ERROR expected `,`, or `}`, found keyword `fn`
    //~| ERROR expected identifier, found keyword `fn`
    //~| ERROR cannot find type `dyn` in this scope
}

fn main() {}
