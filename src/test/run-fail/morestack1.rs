// xfail-test
// compile-flags:--stack-growth
// error-pattern:explicit failure
fn getbig(i: int) {
    if i != 0 {
        getbig(i - 1);
    } else {
        fail;
    }
}

fn main() {
    getbig(100000);
}