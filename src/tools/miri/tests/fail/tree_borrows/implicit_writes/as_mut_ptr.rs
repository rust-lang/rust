// This code works in Tree Borrows without implicit writes, but is expected to fail with implicit writes.
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
