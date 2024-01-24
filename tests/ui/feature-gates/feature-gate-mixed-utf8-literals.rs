fn main() {
    _ = b"a¥🦀"; //~ ERROR mixed utf8
    _ = br"a¥🦀"; //~ ERROR mixed utf8
    _ = b"a\u{a5}\u{1f980}"; //~ ERROR mixed utf8
}
