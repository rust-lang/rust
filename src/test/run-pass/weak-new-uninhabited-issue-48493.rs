fn main() {
    enum Void {}
    std::rc::Weak::<Void>::new();
    std::sync::Weak::<Void>::new();
}
