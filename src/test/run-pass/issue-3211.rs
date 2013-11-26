pub fn main() {
    let mut x = 0;
    4096.times(|| x += 1);
    assert_eq!(x, 4096);
    println!("x = {}", x);
}
