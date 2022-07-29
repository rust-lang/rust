struct Bug<A = [(); (let a = (), 1).1]> {
    //~^ `let` expressions are not supported here
    //~| expected expression, found `let` statement
    a: A
}

fn main() {}
