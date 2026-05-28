macro_rules! get_opt {
    ($tgt:expr, $field:ident) => {
        if $tgt.has_$field() {} //~ ERROR expected `{`, found identifier `foo`
    }
}

fn main() {
    get_opt!(bar, foo);
}
