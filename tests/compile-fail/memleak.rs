//error-pattern: the evaluated program leaked memory

fn main() {
    std::mem::forget(Box::new(42));
}
