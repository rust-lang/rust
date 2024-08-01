//@  compile-flags: --edition=2024 -Z unstable-options

macro_rules! m {
    ($e:expr_2021) => { //~ ERROR: fragment specifier `expr_2021` is unstable
        $e
    };
}

fn main() {
    m!(());
}
