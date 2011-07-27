// xfail-stage0
// xfail-stage1
// xfail-stage2
// xfail-stage3
// error-pattern: attempted dynamic environment-capture
fn foo[T]() {
    obj bar(b: T) { }
}
fn main() { }