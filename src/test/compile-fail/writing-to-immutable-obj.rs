// error-pattern: writing to immutable type
obj objy(int x) {
    fn foo() -> () {
        x = 5;
    }
}
fn main() {
}
