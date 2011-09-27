// xfail-test
// error-pattern:mismatched kind

resource r(i: @mutable int) {
    *i = *i + 1;
}

fn main() {
    let a = @mutable 0;
    {
        let i <- ~r(a);
        // Can't copy into here
        let j <- [i];
    }
    log_err *a;
    // this is no good
    assert *a == 2;
}