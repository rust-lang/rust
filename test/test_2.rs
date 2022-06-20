fn easy_neg_abs(x: i32) -> i32 {
    let y = if x > 0 { -1 * x } else { x };
    assert!(y <= 0);
    return y;
}

// fn neg_abs(mut x: i32) -> i32 {
// if x > 0 {
//     x = -1 * x;
// }
// assert!(x <= 0);
// return x;
// }

// fn abs(mut x: i32) -> i32 {
//     if x < 0 {
//         x = -1 * x;
//     }
//     assert!(x >= 0);
//     return x;
// }

fn main() {
    println!("{}", easy_neg_abs(1));
    // println!("{}", neg_abs(1));
    // println!("{}", abs(-2));
}
