//@revisions: stack tree tree_implicit_writes
//@[tree_implicit_writes]compile-flags: -Zmiri-tree-borrows -Zmiri-tree-borrows-implicit-writes
//@[tree]compile-flags: -Zmiri-tree-borrows

fn main() {
    unsafe extern "C" fn write_with_first_arg(ptr_to_val: *mut i32, _hidden_mut_ref_to_val: ...) {
        // Retagging needs to be disabled for arguments
        // within the VaList. Otherwise, this write access
        // will be undefined behavior.
        unsafe {
            *ptr_to_val = 32;
        }
    }

    let mut val: i32 = 0;
    unsafe {
        write_with_first_arg(&raw mut val, &val);
    }
}
