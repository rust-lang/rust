fn main() {
    let mut foo = [1, 2, 3, 4];
    let a = &mut foo[2];
    let b = &mut foo[3]; //~ ERROR cannot borrow `foo[_]` as mutable more than once at a time
    *a = 5;
    *b = 6;
    println!("{:?} {:?}", a, b);
}
