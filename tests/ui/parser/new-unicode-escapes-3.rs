pub fn main() {
    let s1 = "\u{d805}"; //~ ERROR invalid unicode character escape
    let s2 = "\u{ffffff}"; //~ ERROR invalid unicode character escape
}
