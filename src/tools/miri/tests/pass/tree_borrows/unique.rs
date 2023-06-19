//@revisions: default uniq
//@compile-flags: -Zmiri-tree-borrows -Zmiri-tag-gc=0
//@[uniq]compile-flags: -Zmiri-unique-is-unique

#![feature(ptr_internals)]

#[path = "../../utils/mod.rs"]
mod utils;
use utils::macros::*;

use core::ptr::Unique;

// Check general handling of Unique

fn main() {
    unsafe {
        let base = &mut 5u8;
        let alloc_id = alloc_id!(base);
        name!(base);

        let raw = &mut *base as *mut u8;
        name!(raw);

        // We create a `Unique` and expect it to have a fresh tag
        // and uninitialized permissions.
        let uniq = Unique::new_unchecked(raw);

        // With `-Zmiri-unique-is-unique`, `Unique::as_ptr` (which is called by
        // `Vec::as_ptr`) generates pointers with a fresh tag, so to name the actual
        // `base` pointer we care about we have to walk up the tree a bit.
        //
        // We care about naming this specific parent tag because it is the one
        // that stays `Active` during the entire execution, unlike the leaves
        // that will be invalidated the next time `as_ptr` is called.
        //
        // (We name it twice so that we have an indicator in the output of
        // whether we got the distance correct:
        // If the output shows
        //
        //    |- <XYZ: uniq>
        //    '- <XYZ: uniq>
        //
        // then `nth_parent` is not big enough.
        // The correct value for `nth_parent` should be the minimum
        // integer for which the output shows
        //
        //    '- <XYZ: uniq, uniq>
        // )
        //
        // Ultimately we want pointers obtained through independent
        // calls of `as_ptr` to be able to alias, which will probably involve
        // a new permission that allows aliasing when there is no protector.
        let nth_parent = if cfg!(uniq) { 2 } else { 0 };
        name!(uniq.as_ptr()=>nth_parent, "uniq");
        name!(uniq.as_ptr()=>nth_parent, "uniq");
        print_state!(alloc_id);

        // We can activate the Unique and use it mutably.
        *uniq.as_ptr() = 42;
        print_state!(alloc_id);

        // Write through the raw parent disables the Unique
        *raw = 42;
        print_state!(alloc_id);
    }
}
