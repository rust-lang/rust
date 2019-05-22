// Regression test for issue #61033.

macro_rules! test1 {
    ($x:ident, $($tt:tt)*) => { $($tt)+ } //~ERROR this must repeat at least once
}

fn main() {
    test1!(x,);
}
