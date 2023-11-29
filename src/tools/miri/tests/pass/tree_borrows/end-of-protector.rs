// We disable the GC for this test because it would change what is printed.
//@compile-flags: -Zmiri-tree-borrows -Zmiri-provenance-gc=0

// Check that a protector goes back to normal behavior when the function
// returns.
#[path = "../../utils/mod.rs"]
#[macro_use]
mod utils;

fn main() {
    unsafe {
        let data = &mut 0u8;
        name!(data);
        let alloc_id = alloc_id!(data);
        let x = &mut *data;
        name!(x);
        print_state!(alloc_id);
        do_nothing(x); // creates then removes a Protector for a child of x
        let y = &mut *data;
        name!(y);
        print_state!(alloc_id);
        // Invalidates the previous reborrow, but its Protector has been removed.
        *y = 1;
        print_state!(alloc_id);
    }
}

unsafe fn do_nothing(x: &mut u8) {
    name!(x, "callee:x");
    name!(x=>1, "caller:x");
    let alloc_id = alloc_id!(x);
    print_state!(alloc_id);
}
