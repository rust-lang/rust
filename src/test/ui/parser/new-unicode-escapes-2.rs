pub fn main() {
    let s = "\u{260311111111}"; //~ ERROR overlong unicode escape (must have at most 6 hex digits)
}
