//@  compile-flags: --edition=2024 -Z unstable-options

macro_rules! m {
    ($e:expr_2021) => { //~ ERROR: expr_2021 is experimental
        $e
    };
}

fn main() {
    m!(());
}
