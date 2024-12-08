//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
// Test what happens when we read parts of a pointer.
// Related to <https://github.com/rust-lang/rust/issues/69488>.
fn ptr_partial_read() {
    let x = 13;
    let y = &x;
    let z = &y as *const &i32 as *const u8;

    // This just strips provenance, but should work fine otherwise.
    let _val = unsafe { *z };
}

fn transmute_strip_provenance() {
    let r = &mut 42;
    let addr = r as *mut _ as usize;
    let i: usize = unsafe { std::mem::transmute(r) };
    assert_eq!(i, addr);
}

fn main() {
    ptr_partial_read();
    transmute_strip_provenance();
}
