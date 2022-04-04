fn strip_lf(s: &str) -> &str {
    s.strip_suffix(b'\n').unwrap_or(s)
    //~^ ERROR expected a `FnMut<(char,)>` closure, found `u8`
    //~| NOTE expected an `FnMut<(char,)>` closure, found `u8`
    //~| NOTE required by a bound introduced by this call
    //~| HELP the trait `FnMut<(char,)>` is not implemented for `u8`
    //~| HELP the following other types implement trait `Pattern<'a>`:
    //~| NOTE required because of the requirements on the impl of `Pattern<'_>` for `u8`

}

fn main() {}
