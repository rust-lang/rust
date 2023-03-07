// run-rustfix
#![warn(clippy::repeat_once)]
#[allow(unused, clippy::redundant_clone)]
fn main() {
    const N: usize = 1;
    let s = "str";
    let string = "String".to_string();
    let slice = [1; 5];

    let a = [1; 5].repeat(1);
    let b = slice.repeat(1);
    let c = "hello".repeat(N);
    let d = "hi".repeat(1);
    let e = s.repeat(1);
    let f = string.repeat(1);
}
