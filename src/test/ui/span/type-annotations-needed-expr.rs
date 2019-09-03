fn main() {
    let _ = (vec![1,2,3]).into_iter().sum() as f64; //~ ERROR E0282
}
