


// -*- rust -*-
fn main() {
    let i: int = 90;
    while i < 100 {
        log i;
        i = i + 1;
        if i == 95 {
            let v: [int] =
                ~[1, 2, 3, 4, 5]; // we check that it is freed by break

            log "breaking";
            break;
        }
    }
    assert (i == 95);
}
