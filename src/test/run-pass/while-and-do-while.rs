

fn main() {
    let x: int = 10;
    let y: int = 0;
    while y < x { log_full(core::debug, y); #debug("hello"); y = y + 1; }
    do {
        #debug("goodbye");
        x = x - 1;
        log_full(core::debug, x);
    } while x > 0
}
