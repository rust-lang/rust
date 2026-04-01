//@ run-pass
use std::ptr::addr_of;

// When the unsized tail is a `dyn Trait`, its alignments is only dynamically known. This means the
// packed(2) needs to be applied at runtime: the actual alignment of the field is `min(2,
// usual_alignment)`. Here we check that we do this right by comparing size, alignment, and field
// offset before and after unsizing.
fn main() {
    #[repr(C, packed(2))]
    struct Packed<T: ?Sized>(u8, core::mem::ManuallyDrop<T>);

    let p = Packed(0, core::mem::ManuallyDrop::new(1));
    let p: &Packed<usize> = &p;
    let sized = (core::mem::size_of_val(p), core::mem::align_of_val(p));
    let sized_offset = unsafe { addr_of!(p.1).cast::<u8>().offset_from(addr_of!(p.0)) };
    let p: &Packed<dyn Send> = p;
    let un_sized = (core::mem::size_of_val(p), core::mem::align_of_val(p));
    let un_sized_offset = unsafe { addr_of!(p.1).cast::<u8>().offset_from(addr_of!(p.0)) };
    assert_eq!(sized, un_sized);
    assert_eq!(sized_offset, un_sized_offset);
}
