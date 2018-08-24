fn main() {
    let x = "\\\\\
    ";
    assert_eq!(x, r"\\"); // extraneous whitespace stripped
}
