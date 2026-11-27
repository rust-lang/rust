struct Bug<A = [(); (let a = (), 1).1]> {
    //~^ ERROR expected expression, found `let` statement
    a: A
}

fn main() {}
