

tag opt[T] { none; }

fn main() {
    auto x = none[int];
    alt (x) { case (none[int]) { log "hello world"; } }
}