use std::panic::{catch_unwind, AssertUnwindSafe};

fn main() {
    let mut i = 3;
    let _val = catch_unwind(AssertUnwindSafe(|| i -= 2));
    println!("{}", i);
}
