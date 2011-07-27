


// -*- rust -*-
fn f(x: int) -> int {
    // log "in f:";

    log x;
    if x == 1 {
        // log "bottoming out";

        ret 1;
    } else {
        // log "recurring";

        let y: int = x * f(x - 1);
        // log "returned";

        log y;
        ret y;
    }
}

fn main() {
    assert (f(5) == 120);
    // log "all done";

}