macro_rules! get_opt {
    ($tgt:expr, $field:ident) => {
        if $tgt.has_$field() {} //~ ERROR unexpected macro fragment
    }
}

fn main() {
    get_opt!(bar, foo);
}
