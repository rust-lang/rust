// Make sure we find these even with many checks disabled.
//@compile-flags: -Zmiri-disable-alignment-check -Zmiri-disable-stacked-borrows -Zmiri-disable-validation

#[derive(Copy, Clone)]
#[allow(unused)]
enum E {
    A,
    B,
    C,
}

fn cast(ptr: *const E) {
    unsafe {
        let _val = *ptr as u32; //~ERROR: enum value has invalid tag
    }
}

pub fn main() {
    let v = u32::MAX;
    cast(&v as *const u32 as *const E);
}
