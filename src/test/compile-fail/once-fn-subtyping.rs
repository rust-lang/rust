fn main() {
    let f: &once fn() = ||();
    let g: &fn() = f;  //~ ERROR mismatched types
    let h: &fn() = ||();
    let i: &once fn() = h;  // ok
}

