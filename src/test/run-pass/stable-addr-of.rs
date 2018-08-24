// Issue #2040


pub fn main() {
    let foo: isize = 1;
    assert_eq!(&foo as *const isize, &foo as *const isize);
}
