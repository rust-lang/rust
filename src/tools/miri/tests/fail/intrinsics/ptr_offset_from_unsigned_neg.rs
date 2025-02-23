//@normalize-stderr-test: "\d+ < \d+" -> "$$ADDR < $$ADDR"
fn main() {
    let arr = [0u8; 8];
    let ptr1 = arr.as_ptr();
    let ptr2 = ptr1.wrapping_add(4);
    let _val = unsafe { ptr1.offset_from_unsigned(ptr2) }; //~ERROR: first pointer has smaller address than second
}
