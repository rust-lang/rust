// aux-build:type_dependent_lib.rs
// run-pass
// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

extern crate type_dependent_lib;

use type_dependent_lib::*;

fn main() {
    let s = Struct::<42>::new();
    assert_eq!(s.same_ty::<7>(), (42, 7));
    assert_eq!(s.different_ty::<19>(), (42, 19));
    assert_eq!(Struct::<1337>::new().different_ty::<96>(), (1337, 96));
    assert_eq!(
        Struct::<18>::new()
            .we_have_to_go_deeper::<19>()
            .containing_ty::<Option<u32>, 3>(),
        (27, 3),
    );

    let s = Struct::<7>::new();
    assert_eq!(s.foo::<18>(), 18);
}
