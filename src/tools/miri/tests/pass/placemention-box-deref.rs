fn main() {
    let x = Box::new(Box::new(Box::new(0)));
    drop(**x);
    _ = **x;
}
