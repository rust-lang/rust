pub fn main() {
    let mut x = 0;
    do 4096.times {
        x += 1;
    }
    assert_eq!(x, 4096);
    printfln!("x = %u", x);
}
