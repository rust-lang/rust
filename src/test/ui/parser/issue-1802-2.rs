fn log(a: i32, b: i32) {}

fn main() {
    let error = 42;
    log(error, 0b_usize);
    //~^ ERROR no valid digits found for number
    //~| ERROR mismatched types
}
