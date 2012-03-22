

fn main() {
    let mut x: int = 10;
    let mut y: int = 0;
    while y < x { log(debug, y); #debug("hello"); y = y + 1; }
    do {
        #debug("goodbye");
        x = x - 1;
        log(debug, x);
    } while x > 0
}
