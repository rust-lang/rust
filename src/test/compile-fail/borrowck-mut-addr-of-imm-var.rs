fn main() {
    let x: int = 3;
    let y: &mut int = &mut x; //! ERROR taking mut reference to immutable local variable
    *y = 5;
    log (debug, *y);
}
