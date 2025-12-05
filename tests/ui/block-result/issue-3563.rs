trait A {
    fn a(&self) {
        || self.b()
        //~^ ERROR no method named `b` found
    }
}
fn main() {}
