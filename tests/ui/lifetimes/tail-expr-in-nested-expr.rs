//@ edition: 2024

fn main() {
    let _ = { String::new().as_str() }.len();
    //~^ ERROR temporary value dropped while borrowed
}
