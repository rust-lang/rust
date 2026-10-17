//@compile-flags: -Zmiri-tree-borrows -Zmiri-tree-borrows-implicit-writes
// This test is identical to `fail/stacked_borrows/fnentry_invalidation.rs`.
// This test shows that when `-Zmiri-tree-borrows-implicit-writes` is enabled, Tree Borrows behaves more like Stacked Borrows, and the additional write in `fail/tree_borrows/fnentry_invalidation.rs` is not needed to detect / cause UB.

fn main() {
    let mut x = 0i32;
    let z = &mut x as *mut i32;
    x.do_bad();
    unsafe {
        let _oof = *z; //~ ERROR: /read access through .* at .* is forbidden/
    }
}

trait Bad {
    fn do_bad(&mut self) {
        // who knows
    }
}

impl Bad for i32 {}
