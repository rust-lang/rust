fn main() {
    let x: int = 3;
    let y: &mutable int = &mutable x;
    *y = 5;
    log (debug, *y);
}


