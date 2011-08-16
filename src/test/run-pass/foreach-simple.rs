


// -*- rust -*-
fn main() { for each i: int in first_ten() { log "main"; } }

iter first_ten() -> int {
    let i: int = 90;
    while i < 100 { log "first_ten"; log i; put i; i = i + 1; }
}
