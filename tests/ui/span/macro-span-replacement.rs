// check-pass

#![warn(unused)]

macro_rules! m {
    ($a:tt $b:tt) => {
        $b $a; //~ WARN struct `S` is never constructed
    }
}

fn main() {
    m!(S struct);
}
