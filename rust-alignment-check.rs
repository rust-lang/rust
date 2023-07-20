use std::mem::{align_of, size_of};

#[repr(C)]
struct Struct1 {
    a: u8,
    b: u128,
    c: i128,
}

fn main() {
    println!("i128 size {} align {}", size_of::<i128>(), align_of::<i128>());
    println!("u128 size {} align {}", size_of::<u128>(), align_of::<u128>());
    println!("Struct1 size {} align {}", size_of::<Struct1>(), align_of::<Struct1>());
}
