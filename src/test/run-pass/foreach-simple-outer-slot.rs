


// -*- rust -*-
fn main() {
    let int sum = 0;
    for each (int i in first_ten()) { log "main"; log i; sum = sum + i; }
    log "sum";
    log sum;
    assert (sum == 45);
}

iter first_ten() -> int {
    let int i = 0;
    while (i < 10) { log "first_ten"; put i; i = i + 1; }
}