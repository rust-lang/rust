// error-pattern:mismatched kinds for '~' operand
// xfail-test
resource r(i: @mutable int) {
    *i = *i + 1;
}

fn main() {
    let i = @mutable 0;
    {
        let j <- r(i);
        // No no no no no
        let k <- ~j;
    }
    log_err *i;
    assert *i == 2;
}