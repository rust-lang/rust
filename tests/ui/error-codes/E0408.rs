fn main() {
    let x = Some(0);

    match x {
        Some(y) | None => {} //~  ERROR variable `y` is not bound in all patterns
        _ => ()
    }
}
