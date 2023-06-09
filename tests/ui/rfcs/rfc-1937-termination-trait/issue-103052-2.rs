// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![feature(return_position_impl_trait_in_trait)]
#![allow(incomplete_features)]

mod child {
    trait Main {
        fn main() -> impl std::process::Termination;
    }

    struct Something;

    impl Main for () {
        fn main() -> Something {
            //~^ ERROR the trait bound `Something: Termination` is not satisfied
            Something
        }
    }
}

fn main() {}
