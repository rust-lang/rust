fn main() {
    let a = concat_bytes!(b'A', b"BC"); //~ ERROR use of unstable library feature `concat_bytes`
    assert_eq!(a, &[65, 66, 67]);
}
