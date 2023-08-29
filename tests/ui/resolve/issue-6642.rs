struct A;
impl A {
    fn m(&self) {
        fn x() {
            self.m() //~ ERROR can't capture dynamic environment in a fn item
        }
    }
}
fn main() {}
