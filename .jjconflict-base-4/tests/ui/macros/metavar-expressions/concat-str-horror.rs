//@ run-pass
#![feature(macro_metavar_expr_concat)]

macro_rules! format_horror{
    ($arg1:ident, $arg2:expr, $arg3:expr) =>  {
        format!(${concat_str("horror ", $arg1, " horror {} horror {}")}, $arg2, $arg3)
    }
}

fn main(){
    let x = format_horror!(a, "b", "c");
    assert_eq!(x, "horror a horror b horror c");
}
