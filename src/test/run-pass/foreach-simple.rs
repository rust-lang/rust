


// -*- rust -*-
fn main() { for each (int i in first_ten()) { log "main"; } }

iter first_ten() -> int {
    let int i = 90;
    while (i < 100) { log "first_ten"; log i; put i; i = i + 1; }
}