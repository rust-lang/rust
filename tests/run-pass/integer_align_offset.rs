fn main() {
    assert_eq!(2, (2 as *const i8).align_offset(4));
}
