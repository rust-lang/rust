fn main() {
    let mut v = vec![Some("foo"), Some("bar")];
    v.push(v.pop().unwrap()); //~ ERROR cannot borrow
}
