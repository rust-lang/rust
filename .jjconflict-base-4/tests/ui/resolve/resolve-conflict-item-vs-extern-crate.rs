//@ edition:2015
fn std() {}
mod std {}    //~ ERROR the name `std` is defined multiple times

fn main() {
}
