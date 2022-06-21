// fn easy_neg_abs(x: i32) -> i32 {
//     // let y = if x > 0 { -1 * x } else { x };
//     // assert!(y <= 0);
//     // return y;
//     x + 1
// }
// 2147483648

fn minus_one_safe(x: i32) -> i32 {
    if x > 0 {
        return x - 1;
    }
    return x;
}

fn minus_one_unsafe(x: i32) -> i32 {
    return x - 1;
}

fn neg_abs(mut x: i32) -> i32 {
    if x > 0 {
        x = -1 * x;
    }
    return x;
}

fn abs(mut x: i32) -> i32 {
    if x < 0 {
        x = -1 * x;
    }
    return x;
}

fn main() {
    // println!("{}", easy_neg_abs(1));
    println!("{}", minus_one_safe(3));
    println!("{}", minus_one_unsafe(3));
    println!("{}", neg_abs(2));
    println!("{}", abs(-2));
}
