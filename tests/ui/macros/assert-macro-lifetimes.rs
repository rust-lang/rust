//@ check-pass
#[derive(PartialEq, Eq, Hash)]
struct S;
fn main() {
    let foo = std::rc::Rc::new(std::cell::RefCell::new(std::collections::HashMap::<S, S>::new()));
    // Ensure that the lifetimes of the borrow do not leak past the end of `main`.
    assert!(matches!(foo.borrow().get(&S).unwrap(), S))
}
