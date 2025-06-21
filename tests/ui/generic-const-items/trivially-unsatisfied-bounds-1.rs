//@ check-pass
#![feature(generic_const_items)]
#![expect(incomplete_features)]
#![crate_type = "lib"]

const _UNUSED: () = ()
where
    for<'_delay> String: Copy;

pub const PUB: () = ()
where
    for<'_delay> String: Copy;

