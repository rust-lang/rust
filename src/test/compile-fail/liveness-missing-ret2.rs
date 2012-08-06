// error-pattern: not all control paths return a value

fn f() -> int {
    // Make sure typestate doesn't interpreturn this match expression
    // as the function result
    match check true { true => { } };
}

fn main() { }
