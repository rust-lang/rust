//@ check-pass
//@ edition:2021

use std::fmt::Debug;

trait GlobalTrait {}

fn main() {
    struct InsideMain;

    impl InsideMain {
        fn foo() {}
    }

    impl GlobalTrait for InsideMain {}

    impl Debug for InsideMain {
        fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            todo!()
        }
    }

    impl PartialEq<()> for InsideMain {
        fn eq(&self, _: &()) -> bool {
            todo!()
        }
    }
}

fn dyn_weirdness() {
    trait LocalTrait {}
    impl dyn LocalTrait {}
    impl GlobalTrait for dyn LocalTrait {}
}

struct Test;
mod do_not_lint_mod {
    pub trait Tait {}

    impl super::Test {
        fn hugo() {}
    }

    impl Tait for super::Test {}
}

fn bitflags() {
    struct Flags;

    const _: () = {
        impl Flags {}
    };
}

fn bitflags_internal() {
    const _: () = {
        struct InternalFlags;
        impl InternalFlags {}
    };
}
