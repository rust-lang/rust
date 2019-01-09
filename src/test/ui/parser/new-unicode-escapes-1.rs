pub fn main() {
    let s = "\u{2603"; //~ ERROR unterminated unicode escape (needed a `}`)
}
