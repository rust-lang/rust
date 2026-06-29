//@revisions: stack tree tree_implicit_writes
//@[tree_implicit_writes]compile-flags: -Zmiri-tree-borrows -Zmiri-tree-borrows-implicit-writes
//@[tree]compile-flags: -Zmiri-tree-borrows
//@compile-flags: -Zmiri-disable-alignment-check -Cdebug-assertions=no

fn main() {
    let mut x = [0u8; 20];
    let x_ptr: *mut u8 = x.as_mut_ptr();
    // At least one of these is definitely unaligned.
    unsafe {
        *(x_ptr as *mut u64) = 42;
        *(x_ptr.add(1) as *mut u64) = 42;
    }
}
