fn wrapper3<copy T>(i: T, j: int) {
    log_full(core::debug, i);
    log_full(core::debug, j);
    // This is a regression test that the spawn3 thunk to wrapper3
    // correctly finds the value of j
    assert j == 123456789;
}

fn spawn3<copy T>(i: T, j: int) {
    let wrapped = bind wrapper3(i, j);
    wrapped();
}

fn main() {
    spawn3(127u8, 123456789);
}