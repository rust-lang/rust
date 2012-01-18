export ptr_eq;

#[doc(
  brief = "Determine if two shared boxes point to the same object"
)]
pure fn ptr_eq<T>(a: @T, b: @T) -> bool {
    // FIXME: ptr::addr_of
    unsafe {
        let a_ptr: uint = unsafe::reinterpret_cast(a);
        let b_ptr: uint = unsafe::reinterpret_cast(b);
        ret a_ptr == b_ptr;
    }
}

#[test]
fn test() {
    let x = @3;
    let y = @3;
    assert (ptr_eq::<int>(x, x));
    assert (ptr_eq::<int>(y, y));
    assert (!ptr_eq::<int>(x, y));
    assert (!ptr_eq::<int>(y, x));
}
