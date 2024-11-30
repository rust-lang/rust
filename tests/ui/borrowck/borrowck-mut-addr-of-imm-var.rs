//@ check-pass
fn main() {
    let x: isize = 3;
    let y: &mut isize = &mut x; //~ WARNING cannot borrow
    *y = 5;
    println!("{}", *y);
}
