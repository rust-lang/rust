#![feature(ptr_mask)]

fn main() {
    let v: u32 = 0xABCDABCD;
    let ptr: *const u32 = &v;

    // u32 is 4 aligned,
    // so the lower `log2(4) = 2` bits of the address are always 0
    assert_eq!(ptr.addr() & 0b11, 0);

    let tagged_ptr = ptr.map_addr(|a| a | 0b11);
    let tag = tagged_ptr.addr() & 0b11;
    let masked_ptr = tagged_ptr.mask(!0b11);

    assert_eq!(tag, 0b11);
    assert_eq!(unsafe { *masked_ptr }, 0xABCDABCD);
}
