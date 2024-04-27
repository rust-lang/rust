fn main() {
    let _ = match Some(42) { Some(x) => x, None => "" }; //~ ERROR E0308
}
