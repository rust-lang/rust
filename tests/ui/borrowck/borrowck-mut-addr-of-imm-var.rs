fn main() {
    let x: isize = 3;
    let y: &mut isize = &mut x; //~ ERROR cannot borrow
    *y = 5;
    println!("{}", *y);
}
