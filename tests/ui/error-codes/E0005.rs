fn main() {
    let x = Some(1);
    let Some(y) = x; //~ ERROR E0005
}
