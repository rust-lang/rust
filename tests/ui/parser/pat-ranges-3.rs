// Parsing of range patterns

fn main() {
    let 10 ..= 10 + 3 = 12;
    //~^ error: expected a pattern range bound, found an expression

    let 10 - 3 ..= 10 = 8;
    //~^ error: expected a pattern range bound, found an expression
}
