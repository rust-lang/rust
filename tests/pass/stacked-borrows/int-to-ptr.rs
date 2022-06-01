fn main() {
    ref_raw_int_raw();
}

// Just to make sure that casting a ref to raw, to int and back to raw
// and only then using it works. This rules out ideas like "do escape-to-raw lazily";
// after casting to int and back, we lost the tag that could have let us do that.
fn ref_raw_int_raw() {
    let mut x = 3;
    let xref = &mut x;
    let xraw = xref as *mut i32 as usize as *mut i32;
    assert_eq!(unsafe { *xraw }, 3);
}
