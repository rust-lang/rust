fn main() {
    let var = 10i32;
    let ref string: String = var; //~ ERROR mismatched types [E0308]
}
