// compile-flags: -Z parse-only -Z continue-parse-after-error

fn main() {
    /// document
    //~^ ERROR found a documentation comment that doesn't document anything
    //~| HELP maybe a comment was intended
}
