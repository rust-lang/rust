//@ check-pass
//@ compile-flags: -Znext-solver

fn main() {
    let x: Box<dyn Iterator<Item = ()>> = Box::new(std::iter::empty());
}
