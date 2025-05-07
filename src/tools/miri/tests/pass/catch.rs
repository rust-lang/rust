use std::panic::{AssertUnwindSafe, catch_unwind};

fn main() {
    let mut i = 3;
    let _val = catch_unwind(AssertUnwindSafe(|| i -= 2));
    println!("{}", i);
}
