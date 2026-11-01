//@compile-flags: -Zmiri-tree-borrows -Zmiri-permissive-provenance

#[path = "../../../utils/mod.rs"]
mod utils;

/// Checks that the garbage collector doesn't remove any exposed tags.
fn main() {
    let mut _x: u32 = 4;
    let int = {
        let y = &_x;
        y as *const u32 as usize
    };
    // If y wasn't exposed, this would gc it.
    utils::run_provenance_gc();
    // This should disable y.
    _x = 5;
    let wild = int as *const u32;

    let _fail = unsafe { *wild }; //~ ERROR: /read access through <wildcard> at .* is forbidden/
}
