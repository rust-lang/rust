pub fn main() {
    let mut x = 0;
    for _ in range(0, 4096) { x += 1; }
    assert_eq!(x, 4096);
    println!("x = {}", x);
}
