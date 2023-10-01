//@no-rustfix: overlapping suggestions
#![warn(clippy::needless_for_each)]
#![allow(clippy::needless_return, clippy::uninlined_format_args)]

fn main() {
    let v: Vec<i32> = Vec::new();
    // This is unfixable because the closure includes `return`.
    v.iter().for_each(|v| {
        //~^ ERROR: needless use of `for_each`
        //~| NOTE: `-D clippy::needless-for-each` implied by `-D warnings`
        if *v == 10 {
            return;
        } else {
            println!("{}", v);
        }
    });
}
