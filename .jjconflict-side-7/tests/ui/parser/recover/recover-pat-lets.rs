fn main() {
    let x = Some(2);

    let x.expect("foo");
    //~^ error: expected a pattern, found an expression

    let x.unwrap(): u32;
    //~^ error: expected a pattern, found an expression

    let x[0] = 1;
    //~^ error: expected a pattern, found an expression

    let Some(1 + 1) = x else { //~ error: expected a pattern, found an expression
        return;
    };

    if let Some(1 + 1) = x { //~ error: expected a pattern, found an expression
        return;
    }
}
