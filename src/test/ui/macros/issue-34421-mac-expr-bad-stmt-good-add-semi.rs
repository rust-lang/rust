// check-pass

macro_rules! make_item {
    ($a:ident) => {
        struct $a;
    };
}

fn a() {
    make_item!(A)
}
fn b() {
    make_item!(B)
}

fn main() {}
