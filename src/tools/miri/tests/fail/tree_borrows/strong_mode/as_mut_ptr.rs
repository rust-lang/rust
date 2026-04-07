// This code no longer works using the strong mode in tree borrows.
// This code tests that. The passing version is in `pass/tree_borrows/strong_mode/as_mut_ptr.rs`.
//@compile-flags: -Zmiri-tree-borrows

fn main() {
    let mut x = ["one", "two", "three"];

    let ptr = std::ptr::from_mut(&mut x);
    let a = unsafe { &mut *ptr };
    let b = unsafe { &mut *ptr };

    let _c = as_mut_ptr(a);
    println!("{:?}", *b); //~ ERROR: /Undefined Behavior: reborrow through .* at .* is forbidden/
}

pub const fn as_mut_ptr(x: &mut [&str; 3]) -> *mut str {
    x as *mut [&str] as *mut str
}
