fn main() {
    let f: &fn() = ||();
    let g: &once fn() = f;  //~ ERROR mismatched types
}

