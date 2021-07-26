fn main() {
    assert_eq!(1, 2) //~ ERROR: expected `;`
    assert_eq!(3, 4) //~ ERROR: expected `;`
    println!("hello");
}
