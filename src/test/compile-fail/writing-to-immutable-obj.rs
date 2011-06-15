// xfail-stage0
// error-pattern:assigning to immutable obj field
obj objy(int x) {
    fn foo() -> () {
        x = 5;
    }
}
fn main() {
}
