//@ check-pass
fn main() {
    let v: Vec<isize> = vec![1, 2, 3];
    v[1] = 4; //~ WARNING cannot borrow `v` as mutable, as it is not declared as mutable
}
