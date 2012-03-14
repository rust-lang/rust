#[doc = "Operations on shared box types"];

export ptr_eq;

pure fn ptr_eq<T>(a: @T, b: @T) -> bool unchecked {
    #[doc = "Determine if two shared boxes point to the same object"];
    ptr::addr_of(*a) == ptr::addr_of(*b)
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
