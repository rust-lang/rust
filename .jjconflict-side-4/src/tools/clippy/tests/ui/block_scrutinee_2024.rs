//@ edition: 2024
//@ check-pass
#![warn(clippy::block_scrutinee)]
#![allow(clippy::blocks_in_conditions)]

fn my_function() -> Option<i32> {
    Some(1)
}

fn main() {
    // This should NOT trigger the lint on the 2024 edition
    if let Some(x) = { my_function() } {
        let _ = x;
    }

    // This should NOT trigger the lint on the 2024 edition
    match { my_function() } {
        Some(1) => println!("one"),
        Some(_) => println!("other"),
        None => println!("none"),
    }

    // This should NOT trigger the lint on the 2024 edition
    let mut v = vec![1, 2, 3];
    while let Some(x) = { v.pop() } {
        let _ = x;
    }
}
