fn main() {
    let x = Some(2);

    let Some(1 + 1) = x else { //~ error: expected a pattern, found an expression
        return;
    };
}
