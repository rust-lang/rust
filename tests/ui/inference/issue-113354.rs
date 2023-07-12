//run-rustfix
fn main() {
    let _ = || { while Some(_) = Some(1) { } }; //~ ERROR mismatched types
}
