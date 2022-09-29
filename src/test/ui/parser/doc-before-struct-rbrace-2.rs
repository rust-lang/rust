struct X {
    a: u8 /// document
    //~^ ERROR found a documentation comment that doesn't document anything
    //~| HELP if a comment was intended use `//`
}

fn main() {
    let y = X {a: 1};
}
