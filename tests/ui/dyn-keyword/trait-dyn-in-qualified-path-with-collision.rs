trait Trait {
    fn function(&self) {}
}
impl dyn Trait {
    fn function(&self) {}
}
impl Trait for () {}
fn main() {
    <dyn Trait>::function(&());
    //~^ ERROR multiple applicable items in scope
}
