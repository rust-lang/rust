//@ known-bug: #123955
//@ compile-flags: -Clto -Zvirtual-function-elimination
//@ only-x86_64
pub fn main() {
    _ = Box::new(()) as Box<dyn Send>;
}
