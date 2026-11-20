// This test is meant to test that we can have a stable item in an unstable module, and that
// calling that item through the unstable module is unstable, but that re-exporting it from another
// crate in a stable module is fine.
//
// This is necessary to support moving items from `std` into `core` or `alloc` unstably while still
// exporting the original stable interface in `std`, such as moving `Error` into `core`.
//
//@ aux-build:stable-in-unstable-core.rs
//@ aux-build:stable-in-unstable-std.rs
#![crate_type = "lib"]

extern crate stable_in_unstable_core;
extern crate stable_in_unstable_std;

mod isolated1 {
    use stable_in_unstable_core::new_unstable_module; //~ ERROR use of unstable library feature `unstable_test_feature`
    use stable_in_unstable_core::new_unstable_module::OldTrait; //~ ERROR use of unstable library feature `unstable_test_feature`
}

mod isolated2 {
    use stable_in_unstable_std::old_stable_module::OldTrait;

    struct LocalType;

    impl OldTrait for LocalType {}
}

mod isolated3 {
    use stable_in_unstable_core::new_unstable_module::OldTrait; //~ ERROR use of unstable library feature `unstable_test_feature`

    struct LocalType;

    impl OldTrait for LocalType {}
}

mod isolated4 {
    struct LocalType;

    impl stable_in_unstable_core::new_unstable_module::OldTrait for LocalType {} //~ ERROR use of unstable library feature `unstable_test_feature`
}

mod isolated5 {
    struct LocalType;

    impl stable_in_unstable_std::old_stable_module::OldTrait for LocalType {}
}

mod isolated6 {
    use stable_in_unstable_core::new_unstable_module::{OldTrait}; //~ ERROR use of unstable library feature `unstable_test_feature`
}

mod isolated7 {
    use stable_in_unstable_core::new_unstable_module::*; //~ ERROR use of unstable library feature `unstable_test_feature`
}
