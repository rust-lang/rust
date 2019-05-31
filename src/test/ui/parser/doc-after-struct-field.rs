struct X {
    a: u8 /** document a */,
    //~^ ERROR found a documentation comment that doesn't document anything
    //~| HELP maybe a comment was intended
}

struct Y {
    a: u8 /// document a
    //~^ ERROR found a documentation comment that doesn't document anything
    //~| HELP maybe a comment was intended
}

fn main() {
    let x = X { a: 1 };
    let y = Y { a: 1 };
}
