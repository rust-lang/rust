//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
fn main() {
    unsafe {
        let mut x = 0;
        let ptr = std::ptr::addr_of_mut!(x);
        let frozen = &*ptr;
        let _val = *frozen;
        x = 1;
        let _val = *frozen;
        //~[stack]^ ERROR: /read access .* tag does not exist in the borrow stack/
        //~[tree]| ERROR: /read access through .* is forbidden/
        let _val = x; // silence warning
    }
}
