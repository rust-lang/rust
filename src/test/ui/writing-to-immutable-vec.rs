fn main() {
    let v: Vec<isize> = vec![1, 2, 3];
    v[1] = 4; //~ ERROR cannot borrow immutable local variable `v` as mutable
}
