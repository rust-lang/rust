// error-pattern:fail
// xfail-test

resource r(i: int) {
    // Double-fail!!
    fail;
}

fn main() {
    @0;
    let r <- r(0);
    fail;
}