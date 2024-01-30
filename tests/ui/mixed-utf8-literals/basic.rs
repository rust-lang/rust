// check-pass

#![feature(mixed_utf8_literals)]

fn main() {
    b"a¥🦀";
    b"é";
    b"字";

    br"a¥🦀";
    br"é";
    br##"é"##;

    b"\u{a66e}";
    b"a\u{a5}\u{1f980}";
    b"\u{a4a4}";

    b"hello\xff我叫\u{1F980}";
}
