fn main() {
    let x = Some(3);
    let y = vec![1, 2];
    if let Some(y) = x {
        y.push(y); //~ ERROR E0599
    }
}
