// xfail-test
// error-pattern:mismatched kinds for tag parameter
resource r(i: @mutable int) {
    *i = *i + 1;
}

tag t {
    t0(r);
}

fn main() {
    let i = @mutable 0;
    {
        let j <- r(i);
        // No no no no no
        let k <- t0(j);
    }
    log_err *i;
    assert *i == 2;
}