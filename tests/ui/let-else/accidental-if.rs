fn main() {
    let x = Some(123);
    if let Some(y) = x else { //~ ERROR this `if` expression is missing a block
        return;
    };
}
