//@ known-bug: #156101
fn main() {
    format_args!(concat!("𐏿", "{f:?#}"));
}
