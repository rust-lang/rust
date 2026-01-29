fn main() {
    let x = Some(42);
    if let Some(_) = x
        && Some(x) = x //~^ ERROR expected expression, found `let` statement
    {}
}
