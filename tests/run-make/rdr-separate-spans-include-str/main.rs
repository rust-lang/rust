#[macro_use]
extern crate dep;

fn main() {
    let data = dep_data!();
    assert!(data.contains("hello from dep data"));

    let bytes = dep_bytes!();
    assert!(bytes.starts_with(b"hello bytes data"));
}
