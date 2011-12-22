


// -*- rust -*-
fn f(x: int) -> int {
    // #debug("in f:");

    log_full(core::debug, x);
    if x == 1 {
        // #debug("bottoming out");

        ret 1;
    } else {
        // #debug("recurring");

        let y: int = x * f(x - 1);
        // #debug("returned");

        log_full(core::debug, y);
        ret y;
    }
}

fn main() {
    assert (f(5) == 120);
    // #debug("all done");

}
