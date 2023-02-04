#[deny(unused)]
fn main() {
    let arr = [0; 10];
    let _ = arr[(0)]; //~ ERROR unnecessary parentheses around index expression
    let _ = arr[{0}]; //~ ERROR unnecessary braces around index expression
    let _ = arr[1 + (0)];
    let _ = arr[{ let x = 0; x }];
}
