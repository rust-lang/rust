// run-pass
macro_rules! foo {
    ($t:ty; $p:path;) => {}
}

fn main() {
    foo!(i32; i32;);
}
