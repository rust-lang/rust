//@ check-pass

macro_rules! foo {
    ($a:literal) => {
        bar!($a)
    };
}

macro_rules! bar {
    ($b:literal) => {};
}

fn main() {
    foo!(-2);
    bar!(-2);
}
