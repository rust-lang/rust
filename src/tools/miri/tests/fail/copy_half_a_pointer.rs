//@normalize-stderr-test: "\+0x[48]" -> "+HALF_PTR"
#![allow(dead_code)]

// We use packed structs to get around alignment restrictions
#[repr(packed)]
struct Data {
    pad: u8,
    ptr: &'static i32,
}

static G: i32 = 0;

fn main() {
    let mut d = Data { pad: 0, ptr: &G };

    // Get a pointer to the beginning of the Data struct (one u8 byte, then the pointer bytes).
    let d_alias = &mut d as *mut _ as *mut *const u8;
    unsafe {
        let _x = d_alias.read_unaligned(); //~ERROR: unable to copy parts of a pointer
    }
}
