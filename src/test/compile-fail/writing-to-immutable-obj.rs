// error-pattern:assigning to immutable object field
obj objy(x: int) {
    fn foo() { x = 5; }
}
fn main() { }
