fn main() {
    auto x = true;
    if x {
        auto i = 10;
        while i > 0 { i -= 1; }
    }
    alt x {
        true { log "right"; }
        false { log "wrong"; }
    }
}