fn main() {
    let x = std::env::args(); //~ ERROR miri does not support program arguments
    assert_eq!(x.count(), 1);
}
