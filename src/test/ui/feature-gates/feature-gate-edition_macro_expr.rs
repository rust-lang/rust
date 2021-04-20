// feature gate test for edition_macro_expr

macro_rules! mac {
    ($x:expr2015) => {}; //~ERROR `expr2015` and `expr202x` are unstable
    ($x:expr202x) => {}; //~ERROR `expr2015` and `expr202x` are unstable
}

fn main() {}
