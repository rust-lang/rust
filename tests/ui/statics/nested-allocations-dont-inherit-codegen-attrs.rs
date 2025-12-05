//@ build-pass

// Make sure that the nested static allocation for `FOO` doesn't inherit `no_mangle`.
#[no_mangle]
pub static mut FOO: &mut [i32] = &mut [42];

// Make sure that the nested static allocation for `BAR` doesn't inherit `export_name`.
#[export_name = "BAR_"]
pub static mut BAR: &mut [i32] = &mut [42];

fn main() {}
