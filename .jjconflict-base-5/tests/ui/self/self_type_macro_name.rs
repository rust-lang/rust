pub fn main() {
    match 15 {
        Self!() => (),
        //~^ ERROR cannot find macro `Self` in this scope
    }
}
