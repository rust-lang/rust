// Check that we don't evaluate the initializer of free const items if
// the latter have "impossible" (trivially unsatisfied) predicates.
//@ check-pass

#![feature(generic_const_items)]
#![expect(incomplete_features)]
#![crate_type = "lib"]

const _UNUSED: () = panic!()
where
    for<'_delay> String: Copy;

// Check *public* const items specifically to exercise reachability analysis which normally
// evaluates const initializers to look for function pointers in the final const value.
pub const PUBLIC: () = panic!()
where
    for<'_delay> String: Copy;

const REGION_POLY<'a>: () = panic!()
where
    for<'_delay> [&'a ()]: Sized;
