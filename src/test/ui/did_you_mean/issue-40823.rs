fn main() {
    let mut buf = &[1, 2, 3, 4];
    buf.iter_mut(); //~ ERROR cannot borrow immutable borrowed content
}
