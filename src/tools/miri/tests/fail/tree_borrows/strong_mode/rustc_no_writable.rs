// This code tests that the `#[rustc_no_writable]` attribute and the `-Zno-writable` flag have the desired effect.
// With them, this test should pass, which is tested in `pass/tree_borrows/strong_mode/rustc_no_writable.rs` and `pass/tree_borrows/strong_mode/no_writable.rs`.
//@compile-flags: -Zmiri-tree-borrows

fn main() {
    let mut x = 42;

    let ptr = std::ptr::from_mut(&mut x);
    let a = unsafe { &mut *ptr };
    let b = unsafe { &mut *ptr };

    let _c = foo(a);
    println!("{:?}", *b); //~ ERROR: /Undefined Behavior: reborrow through .* at .* is forbidden/
}

pub fn foo(_x: &mut i32) {}
