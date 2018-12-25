trait A {
    fn a(&self) {
        || self.b()
        //~^ ERROR no method named `b` found for type `&Self` in the current scope
    }
}
fn main() {}
