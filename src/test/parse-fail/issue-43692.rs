// compile-flags: -Z parse-only

fn main() {
    '\u{_10FFFF}'; //~ ERROR invalid start of unicode escape
}
