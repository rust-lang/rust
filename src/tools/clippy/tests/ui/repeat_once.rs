#![warn(clippy::repeat_once)]
#[allow(unused, clippy::redundant_clone)]
fn main() {
    const N: usize = 1;
    let s = "str";
    let string = "String".to_string();
    let slice = [1; 5];

    let a = [1; 5].repeat(1);
    //~^ repeat_once
    let b = slice.repeat(1);
    //~^ repeat_once
    let c = "hello".repeat(N);
    //~^ repeat_once
    let d = "hi".repeat(1);
    //~^ repeat_once
    let e = s.repeat(1);
    //~^ repeat_once
    let f = string.repeat(1);
    //~^ repeat_once
}
