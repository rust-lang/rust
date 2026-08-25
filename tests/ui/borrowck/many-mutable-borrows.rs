fn main() {
    let v = Vec::new(); //~ ERROR cannot borrow `v` as mutable, as it is not declared as mutable
    v.push(0);
    v.push(0);
    v.push(0);
    v.push(0);
    v.push(0);
    v.push(0);
    v.push(0);
    v.push(0);
    v.push(0);
    v.push(0);
    v.push(0);
    v.push(0);
    v.push(0);
    v.push(0);
    v.push(0);
}
