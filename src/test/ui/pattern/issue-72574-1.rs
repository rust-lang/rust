fn main() {
    let x = (1, 2, 3);
    match x {
        (_a, _x @ ..) => {}
        _ => {}
    }
}
//~^^^^ ERROR `_x @` is not allowed in a tuple
//~| ERROR: `..` patterns are not allowed here
//~| ERROR: mismatched types
