macro_rules! get_opt {
    ($tgt:expr, $field:ident) => {
        if $tgt.has_$field() {}
    }
}

fn main() {
    get_opt!(bar, foo);
    //~^ ERROR expected `{`, found `foo`
}
