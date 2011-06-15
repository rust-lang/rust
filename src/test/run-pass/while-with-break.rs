


// -*- rust -*-
fn main() {
    let int i = 90;
    while (i < 100) {
        log i;
        i = i + 1;
        if (i == 95) {
            let vec[int] v =
                [1, 2, 3, 4, 5]; // we check that it is freed by break

            log "breaking";
            break;
        }
    }
    assert (i == 95);
}