//@ edition: 2024
//@ compile-flags: -Zunstable-options

fn main() {
    let _ = { String::new().as_str() }.len();
    //~^ ERROR temporary value dropped while borrowed
}
