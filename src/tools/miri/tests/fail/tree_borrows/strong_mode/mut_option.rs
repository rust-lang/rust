// This tests an interesting edge case for the writable attribute. Even though `_x` is not a mutable borrow, the underlying implementation still behaves like a mutable borrow. Thus, the new UB introduced by the writable attribute also affects this case. This tests shows that behavior.
// A matching test that shows the old behavior is in `pass/tree_borrows/strong_mode/mut_option.rs`. The only difference is the addition of the `#[rustc_no_writable]` attribute.
//@compile-flags: -Zmiri-tree-borrows

fn main() {
    let mut x = 42;

    let ptr = std::ptr::from_mut(&mut x);
    let a = unsafe { &mut *ptr };
    let b = unsafe { &mut *ptr };

    let _c = foo(Some(a));
    println!("{:?}", *b); //~ ERROR: /Undefined Behavior: reborrow through .* at .* is forbidden/
}

// Even though `_x` is not a mutable borrow, the writable attribute is still inserted in LLVM (due to the implementation of `Option``) and thus a write is inserted in the semantics.
fn foo(_x: Option<&mut i32>) {}
