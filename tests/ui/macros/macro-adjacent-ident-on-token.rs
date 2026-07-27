//! Regression test for <https://github.com/rust-lang/rust/issues/39848>.
//! This used to ice on `IllFormedSpan`.

macro_rules! get_opt {
    ($tgt:expr, $field:ident) => {
        if $tgt.has_$field() {} //~ ERROR expected `{`, found identifier `foo`
    }
}

fn main() {
    get_opt!(bar, foo);
}
