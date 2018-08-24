pub fn main() {
    let a: String = "hello".to_string();
    let b: String = "world".to_string();
    let s: String = format!("{}{}", a, b);
    println!("{}", s.clone());
    assert_eq!(s.as_bytes()[9], 'd' as u8);
}
