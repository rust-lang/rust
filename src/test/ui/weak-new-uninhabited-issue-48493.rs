// run-pass

fn main() {
    enum Void {}
    let _ = std::rc::Weak::<Void>::new();
    let _ = std::sync::Weak::<Void>::new();
}
