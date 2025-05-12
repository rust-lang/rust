// Ensure we give the right args when we suggest calling a closure.

fn main() {
    let x = |a: i32, b: i32| a + b;
    let y: i32 = x;
    //~^ ERROR mismatched types
}
