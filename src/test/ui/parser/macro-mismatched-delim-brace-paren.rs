macro_rules! foo { ($($tt:tt)*) => () }

fn main() {
    foo! {
        bar, "baz", 1, 2.0
    ) //~ ERROR incorrect close delimiter
}
