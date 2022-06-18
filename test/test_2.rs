fn neg_abs(mut x: i32) -> i32 {
    if x > 0 {
        x = -1 * x;
    }
    assert!(x <= 0);
    return x;
    // let y = if x > 0 {
    //     -1 * x
    // } else {
    //     x
    // };
    // assert!(y <= 0);
    // return y;
}

fn main() {
    println!("{}", neg_abs(13));
}
