// error-pattern:index out of bounds

use std::mem;

fn main() {

    // This should cause a bounds-check panic, but may not if we do our
    // bounds checking by comparing the scaled index to the vector's
    // address-bounds, since we've scaled the index to wrap around to the
    // address of the 0th cell in the array (even though the index is
    // huge).

    let x = vec![1_usize, 2_usize, 3_usize];

    let base = x.as_ptr() as usize;
    let idx = base / mem::size_of::<usize>();
    println!("ov1 base = 0x{:x}", base);
    println!("ov1 idx = 0x{:x}", idx);
    println!("ov1 sizeof::<usize>() = 0x{:x}", mem::size_of::<usize>());
    println!("ov1 idx * sizeof::<usize>() = 0x{:x}",
             idx * mem::size_of::<usize>());

    // This should panic.
    println!("ov1 0x{:x}", x[idx]);
}
