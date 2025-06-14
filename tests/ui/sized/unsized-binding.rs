fn main() {
    let x = *""; //~ ERROR E0277
    drop(x);
    drop(x);
}
