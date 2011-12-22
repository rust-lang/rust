

fn main() {
    let a: int = 10;
    log_full(core::debug, a);
    assert (a * (a - 1) == 90);
}
