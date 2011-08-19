// error-pattern: unresolved name: base
type base =
    obj {
        fn foo();
    };
obj derived() {
    fn foo() { }
    fn bar() { }
}
fn main() { let d: derived = derived(); let b: base = base(d); }
