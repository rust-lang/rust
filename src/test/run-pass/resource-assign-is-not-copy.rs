resource r(i: @mutable int) { *i += 1; }

fn main() {
    let i = @mutable 0;
    // Even though these look like copies, they are guaranteed not to be
    {
        let a = r(i);
        let b = (a, 10);
        let (c, _d) = b;
        log_full(core::debug, c);
    }
    assert *i == 1;
}
