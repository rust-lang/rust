

fn main() {
    let x: int = 10;
    let y: int = 0;
    while y < x { log y; log "hello"; y = y + 1; }
    do  { log "goodbye"; x = x - 1; log x; } while x > 0
}
