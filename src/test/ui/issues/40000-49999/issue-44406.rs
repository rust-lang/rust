macro_rules! foo {
    ($rest: tt) => {
        bar(baz: $rest)
    }
}

fn main() {
    foo!(true); //~ ERROR expected type, found keyword
    //~^ ERROR expected identifier, found keyword
}
