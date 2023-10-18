// Regression test for #85794

struct Baz {
    inner : dyn fn ()
    //~^ ERROR expected `,`, or `}`, found keyword `fn`
    //~| ERROR expected identifier, found keyword `fn`
}

fn main() {}
