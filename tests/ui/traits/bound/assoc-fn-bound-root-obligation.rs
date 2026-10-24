fn strip_lf(s: &str) -> &str {
    s.strip_suffix(b'\n').unwrap_or(s)
    //~^ ERROR the trait bound `u8: Pattern<str>` is not satisfied
    //~| NOTE required by a bound introduced by this call
    //~| NOTE the trait `FnMut(char)` is not implemented for `u8`
    //~| HELP the following other types implement trait `Pattern<H>`:
    //~| NOTE required for `u8` to implement `Pattern<str>`
    //~| NOTE required by a bound in `core::str::<impl str>::strip_suffix`
}

fn main() {}
