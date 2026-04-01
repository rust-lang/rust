//@ run-pass
//@ compile-flags: -C debug-assertions

#[repr(packed)]
struct Misaligner {
    _head: u8,
    tail: u64,
}

fn main() {
    let memory = [Misaligner { _head: 0, tail: 0}, Misaligner { _head: 0, tail: 0}];
    // Test that we can use addr_of! to get the address of a packed member which according to its
    // type is not aligned, but because it is a projection from a packed type is a valid place.
    let ptr0 = std::ptr::addr_of!(memory[0].tail);
    let ptr1 = std::ptr::addr_of!(memory[0].tail);
    // Even if ptr0 happens to be aligned by chance, ptr1 is not.
    assert!(!ptr0.is_aligned() || !ptr1.is_aligned());

    // And also test that we can get the addr of a packed struct then do a member read from it.
    unsafe {
        let ptr = std::ptr::addr_of!(memory[0]);
        let _tail = (*ptr).tail;

        let ptr = std::ptr::addr_of!(memory[1]);
        let _tail = (*ptr).tail;
    }
}
