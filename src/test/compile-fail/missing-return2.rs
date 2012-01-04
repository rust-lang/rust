// error-pattern: not all control paths return a value

fn f() -> int {
    // Make sure typestate doesn't interpret this alt expression
    // as the function result
    alt true { true { } };
}

fn main() { }
