// error-pattern: unresolved name
obj oT() {
    fn get() -> int { ret 3; }
    fn foo() { let c = get(); }
}
fn main() { }
