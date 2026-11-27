fn a(i: i32) -> i32 {
    //~^ ERROR mismatched types
    let j = 2i32;
}

fn b(i: i32, j: i32) -> i32 {}
//~^ ERROR mismatched types

fn main() {}
