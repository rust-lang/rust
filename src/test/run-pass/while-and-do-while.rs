

fn main() {
    let int x = 10;
    let int y = 0;
    while (y < x) { log y; log "hello"; y = y + 1; }
    do  { log "goodbye"; x = x - 1; log x; } while (x > 0)
}