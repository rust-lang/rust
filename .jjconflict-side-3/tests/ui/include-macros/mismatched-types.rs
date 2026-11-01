fn main() {
    let b: &[u8] = include_str!("file.txt");    //~ ERROR mismatched types
    let s: &str = include_bytes!("file.txt");   //~ ERROR mismatched types
}
