#![crate_type = "lib"]

#[unsafe(no_mangle)]
pub extern "C" fn eszett() -> i8 {
    42
}

#[unsafe(no_mangle)]
pub extern "C" fn beta() -> u32 {
    1
}

fn main() {}
