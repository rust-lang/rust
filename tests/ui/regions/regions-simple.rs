// run-pass
pub fn main() {
    let mut x: isize = 3;
    let y: &mut isize = &mut x;
    *y = 5;
    println!("{}", *y);
}
