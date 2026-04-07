// This code tests that the `-Zno-writable` flag has the desired effect.
// Without it, this test should fail, which is tested in `fail/tree_borrows/strong_mode/rustc_no_writable.rs`.
//@compile-flags: -Zmiri-tree-borrows -Zno-writable

fn main() {
    let mut x = 42;

    let ptr = std::ptr::from_mut(&mut x);
    let a = unsafe { &mut *ptr };
    let b = unsafe { &mut *ptr };

    let _c = foo(a);
    println!("{:?}", *b);
}

pub fn foo(_x: &mut i32) {}
