// error-pattern:fail
// xfail-test

resource r(i: int) {
    // What happens when destructors throw?
    fail;
}

fn main() {
    @0;
    let r <- r(0);
}