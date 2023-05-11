macro_rules! foo {
    ($rest: tt) => {
        bar(baz: $rest) //~ ERROR invalid `struct` delimiters or `fn` call arguments
    }
}

fn main() {
    foo!(true);
}
