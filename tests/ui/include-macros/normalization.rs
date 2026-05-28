//@ run-pass

fn main() {
    assert_eq!(
        &include_bytes!("data.bin")[..],
        &b"\xEF\xBB\xBFThis file starts with BOM.\r\nLines are separated by \\r\\n.\r\n"[..],
    );
    assert_eq!(
        include_str!("data.bin"),
        "\u{FEFF}This file starts with BOM.\r\nLines are separated by \\r\\n.\r\n",
    );
}
