


// -*- rust -*-
fn main() {
    let sum: int = 0;
    first_ten {|i| log "main"; log i; sum = sum + i; };
    log "sum";
    log sum;
    assert (sum == 45);
}

fn first_ten(it: block(int)) {
    let i: int = 0;
    while i < 10 { log "first_ten"; it(i); i = i + 1; }
}
