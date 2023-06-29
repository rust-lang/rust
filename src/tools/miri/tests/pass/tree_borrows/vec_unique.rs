//@revisions: default uniq
//@compile-flags: -Zmiri-tree-borrows -Zmiri-tag-gc=0
//@[uniq]compile-flags: -Zmiri-unique-is-unique

#![feature(vec_into_raw_parts)]

#[path = "../../utils/mod.rs"]
mod utils;
use utils::macros::*;

// Check general handling of `Unique`:
// there is no *explicit* `Unique` being used here, but there is one
// hidden a few layers inside `Vec` that should be reflected in the tree structure.

fn main() {
    unsafe {
        let base = vec![0u8, 1];
        let alloc_id = alloc_id!(base.as_ptr());

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
        name!(base.as_ptr()=>nth_parent);
        name!(base.as_ptr()=>nth_parent);

        // Destruct the `Vec`
        let (ptr, len, cap) = base.into_raw_parts();

        // Expect this to be again the same pointer as the one obtained from `as_ptr`.
        // Under `-Zmiri-unique-is-unique`, this will be a strict child.
        name!(ptr, "raw_parts.0");

        // This is where the presence of `Unique` has implications,
        // because there will be a reborrow here iff the exclusivity of `Unique`
        // is enforced.
        let reconstructed = Vec::from_raw_parts(ptr, len, cap);

        // The `as_ptr` here (twice for the same reason as above) return either
        // the same pointer once more (default) or a strict child (uniq).
        name!(reconstructed.as_ptr()=>nth_parent);
        name!(reconstructed.as_ptr()=>nth_parent);

        print_state!(alloc_id, false);
    }
}
