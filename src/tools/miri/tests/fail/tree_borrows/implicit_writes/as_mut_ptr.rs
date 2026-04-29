// This code no longer works using implicit writes in tree borrows.
// This code tests that. The passing version is in `pass/tree_borrows/implicit_writes/as_mut_ptr.rs`.
//@compile-flags: -Zmiri-tree-borrows -Zmiri-tree-borrows-implicit-writes

fn main() {
    let mut x: [u8; 3] = [1, 2, 3];

    let ptr = std::ptr::from_mut(&mut x);
    let a = unsafe { &mut *ptr };
    let b = unsafe { &mut *ptr };

    let _c = as_mut_ptr(a);
    println!("{:?}", *b); //~ ERROR: /Undefined Behavior: reborrow through .* at .* is forbidden/
}

pub const fn as_mut_ptr(x: &mut [u8; 3]) -> *mut u8 {
    x as *mut [u8] as *mut u8
}
