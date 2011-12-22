


// -*- rust -*-
obj counter(mutable x: int) {
    fn hello() -> int { ret 12345; }
    fn incr() { x = x + 1; }
    fn get() -> int { ret x; }
}

fn main() {
    let y = counter(0);
    assert (y.hello() == 12345);
    log_full(core::debug, y.get());
    y.incr();
    y.incr();
    log_full(core::debug, y.get());
    assert (y.get() == 2);
}
