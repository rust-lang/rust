// error-pattern: not all control paths return a value

fn f() -> int {
    // Make sure typestate doesn't interpreturn this alt expression
    // as the function result
    alt check true { true { } };
}

fn main() { }
