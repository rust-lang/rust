//@ edition:2021
// Check what happens when a const async fn is in the main function (#102796)

fn main() {
    const async fn a() {}
//~^ ERROR functions cannot be both `const` and `async`
}
