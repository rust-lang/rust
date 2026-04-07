//@compile-flags: -Zmiri-tree-borrows -Zmiri-permissive-provenance

#![feature(rustc_attrs)]
#![allow(internal_features)]
#![feature(stmt_expr_attributes)]

/// Checks that wildcard accesses correctly infers the allowed permissions
/// on protected conflicted pointers.
pub fn main() {
    let mut x: u32 = 42;

    let ptr_base = &mut x as *mut u32;
    let ref1 = unsafe { &mut *ptr_base };
    let ref2 = unsafe { &mut *ptr_base };

    let protect = #[rustc_no_writable] // TODO: disable new behavior for now to make test the old thing again. probably have to rewrite this to test the desired behavior instead of disabling new feature
    |arg: &mut u32| {
        // Expose arg.
        let int = arg as *mut u32 as usize;
        let wild = int as *mut u32;

        // Does a foreign read to arg marking it as conflicted and making child
        // writes UB while it's protected.
        let _x = *ref2;

        // The only exposed reference (arg) doesn't allow child writes, so this is UB.
        unsafe { *wild = 4 }; //~ ERROR: /write access through <wildcard> at .* is forbidden/
    };

    protect(ref1);
}
