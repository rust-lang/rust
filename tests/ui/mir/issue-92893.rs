struct Bug<A = [(); (let a = (), 1).1]> {
    //~^ `let` expressions are not supported here
    //~| `let` expressions in this position are unstable [E0658]
    //~| expected expression, found `let` statement
    a: A
}

fn main() {}
