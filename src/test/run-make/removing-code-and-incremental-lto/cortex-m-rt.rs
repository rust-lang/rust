#![no_std]

#[link_section = ".vector_table.reset_vector"]
#[no_mangle] pub static __RESET_VECTOR: unsafe extern "C" fn() -> ! = Reset;

extern "Rust" { fn main() -> !; }

#[no_mangle] pub unsafe extern "C" fn Reset() -> ! { main() }

#[no_mangle] pub unsafe extern "C" fn DefaultHandler_() -> ! { loop { } }

#[link_section = ".vector_table.exceptions"]
#[no_mangle] pub static __EXCEPTIONS: [usize; 14] = [0; 14];
