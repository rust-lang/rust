// error-pattern:quux
// xfail-test

resource faily_box(_i: @int) {
    // What happens to the box pointer owned by this resource?
    fail "quux";
}

fn main() {
    faily_box(@10);
}
