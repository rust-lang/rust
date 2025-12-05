//@ run-pass
struct NonOrd;

fn main() {
    let _: Box<dyn Iterator<Item = _>> = Box::new(vec![NonOrd].into_iter());
}
