//@ known-bug: #158243
macro_rules! id {
    ($x:expr) => { $x };
}
fn main() {
    |id!(
        (|| {
            use std::ops::Add;
            1.add(3);
        })
    )| {};
}
