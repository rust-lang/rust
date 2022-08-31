fn strip_lf(s: &str) -> &str {
    s.strip_suffix(b'\n').unwrap_or(s)
    //~^ ERROR expected a `FnMut<(char,)>` closure, found `u8`
    //~| NOTE expected an `FnMut<(char,)>` closure, found `u8`
    //~| HELP the trait `FnMut<(char,)>` is not implemented for `u8`
    //~| HELP the following other types implement trait `Pattern<'a>`:
    //~| NOTE required for `u8` to implement `Pattern<'_>`

}

fn main() {}
