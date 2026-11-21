//@revisions: tree tree_implicit_writes
//@[tree_implicit_writes]compile-flags: -Zmiri-tree-borrows-implicit-writes
//@compile-flags: -Zmiri-tree-borrows

// Check that we don't add a protector, as this function takes raw pointers.
fn f(_p: *mut u8, r: *mut u8) {
    unsafe { *r = 1 };
}

fn main() {
    let mut v = 0u8;
    let r = &raw mut v;
    let g: fn(&mut u8, *mut u8) = unsafe { std::mem::transmute(f as fn(*mut u8, *mut u8)) };
    // `g` is seen as receiving a `&mut u8`; if we retag on the caller's types, we introduce a
    // protector that the write inside `f` invalidates, causing undesirable UB.
    g(unsafe { &mut *r }, r);
    assert!(v == 1);
}
