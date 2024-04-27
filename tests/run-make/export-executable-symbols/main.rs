//@ edition:2018

fn main() {}

#[no_mangle]
pub fn exported_symbol() -> i8 {
    42
}
