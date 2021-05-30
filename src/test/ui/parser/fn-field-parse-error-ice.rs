// Regression test for #85794

struct Baz {
    inner : dyn fn ()
    //~^ ERROR expected `,`, or `}`, found keyword `fn`
    //~| ERROR functions are not allowed in struct definitions
    //~| ERROR cannot find type `dyn` in this scope
}

fn main() {}
