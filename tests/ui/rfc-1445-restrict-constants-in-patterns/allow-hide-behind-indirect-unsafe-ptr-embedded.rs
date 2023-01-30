// Test explores how `#[structral_match]` behaves in tandem with
// `*const` and `*mut` pointers.

// run-pass

#![warn(pointer_structural_match)]

struct NoDerive(#[allow(unused_tuple_struct_fields)] i32);

// This impl makes NoDerive irreflexive
// (which doesn't matter here because `<*const T>::eq` won't recur on `T`).
impl PartialEq for NoDerive { fn eq(&self, _: &Self) -> bool { false } }

impl Eq for NoDerive { }

#[derive(PartialEq, Eq)]
struct WrapEmbedded(*const NoDerive);

const WRAP_UNSAFE_EMBEDDED: & &WrapEmbedded = & &WrapEmbedded(std::ptr::null());

fn main() {
    match WRAP_UNSAFE_EMBEDDED {
        WRAP_UNSAFE_EMBEDDED => { println!("WRAP_UNSAFE_EMBEDDED correctly matched itself"); }
        _ => { panic!("WRAP_UNSAFE_EMBEDDED did not match itself"); }
    }
}
