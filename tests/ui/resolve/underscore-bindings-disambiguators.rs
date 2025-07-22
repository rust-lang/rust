// Regression test for issue #144168 where some `_` bindings were incorrectly only allowed once per
// module, failing with "error[E0428]: the name `_` is defined multiple times".

// This weird/complex setup is reduced from `zerocopy-0.8.25` where the issue was encountered.

#![crate_type = "lib"]

macro_rules! impl_for_transmute_from {
    () => {
        const _: () = {};
    };
}

mod impls {
    use super::*;
    impl_for_transmute_from!();
    impl_for_transmute_from!();
    const _: () = todo!(); //~ ERROR: evaluation panicked
    const _: () = todo!(); //~ ERROR: evaluation panicked
    const _: () = todo!(); //~ ERROR: evaluation panicked
    const _: () = todo!(); //~ ERROR: evaluation panicked
    const _: () = todo!(); //~ ERROR: evaluation panicked
}
use X as Y; //~ ERROR: unresolved import
use Z as W; //~ ERROR: unresolved import

const _: () = todo!(); //~ ERROR: evaluation panicked
