fn main() {
    let x: int = 3;
    let y: &mut int = &mut x; //~ ERROR illegal borrow
    *y = 5;
    log (debug, *y);
}
