//@ check-pass

#![warn(unused)]

macro_rules! m {
    ($a:tt $b:tt) => {
        $b $a;
    }
}

fn main() {
    m!(S struct); //~ WARN struct `S` is never constructed
}
