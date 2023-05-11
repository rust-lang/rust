// check-pass
// compile-flags: -Ztrait-solver=next

fn main() {
    let x: Box<dyn Iterator<Item = ()>> = Box::new(std::iter::empty());
}
