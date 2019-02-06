fn empty() -> &'static str {
    ""
}

fn hello() -> &'static str {
    "Hello, world!"
}

fn hello_bytes() -> &'static [u8; 13] {
    b"Hello, world!"
}

fn hello_bytes_fat() -> &'static [u8] {
    b"Hello, world!"
}

fn fat_pointer_on_32_bit() {
    Some(5).expect("foo");
}

fn str_indexing() {
    let mut x = "Hello".to_string();
    let _v = &mut x[..3];  // Test IndexMut on String.
}

fn main() {
    assert_eq!(empty(), "");
    assert_eq!(hello(), "Hello, world!");
    assert_eq!(hello_bytes(), b"Hello, world!");
    assert_eq!(hello_bytes_fat(), b"Hello, world!");

    fat_pointer_on_32_bit(); // Should run without crashing.
    str_indexing();
}
