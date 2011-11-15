// error-pattern: mismatched kind
// xfail-test

resource r(b: bool) {
}

fn main() {
    let i <- [r(true)];
    i += [r(true)];
}