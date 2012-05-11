fn main() {
    let mut x: int = 3;
    let y: &mut int = &mut x;
    *y = 5;
    log (debug, *y);
}


