#![allow(clippy::uninlined_format_args)]

fn main() {}

#[warn(clippy::cognitive_complexity)]
fn cognitive_complexity() {
    //~^ cognitive_complexity
    let x = vec![1, 2, 3];
    for i in x {
        if i == 1 {
            println!("{}", i);
        }
    }
}
