fn log(a: i32, b: i32) {}

fn main() {
    let error = 42;
    log(error, 0b);
    //~^ ERROR no valid digits found for number
}
