// build-pass (FIXME(62277): could be check-pass?)

#![warn(unused)]

macro_rules! m {
    ($a:tt $b:tt) => {
        $b $a; //~ WARN struct is never constructed
    }
}

fn main() {
    m!(S struct);
}
