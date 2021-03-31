#![warn(clippy::needless_for_each)]
#![allow(clippy::needless_return)]

fn main() {
    let v: Vec<i32> = Vec::new();
    // This is unfixable because the closure includes `return`.
    v.iter().for_each(|v| {
        if *v == 10 {
            return;
        } else {
            println!("{}", v);
        }
    });
}
