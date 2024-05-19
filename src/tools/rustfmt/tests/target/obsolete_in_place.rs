// #2953

macro_rules! demo {
    ($a:ident <- $b:expr) => {};
}

fn main() {
    demo!(i <- 0);
}
