


// -*- rust -*-
fn f(int x) -> int {
    // log "in f:";

    log x;
    if (x == 1) {
        // log "bottoming out";

        ret 1;
    } else {
        // log "recurring";

        let int y = x * f(x - 1);
        // log "returned";

        log y;
        ret y;
    }
}

fn main() {
    assert (f(5) == 120);
    // log "all done";

}