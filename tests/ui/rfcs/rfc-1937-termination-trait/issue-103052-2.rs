#![allow(incomplete_features)]

mod child {
    trait Main {
        fn main() -> impl std::process::Termination;
    }

    struct Something;

    impl Main for () {
        fn main() -> Something {
            //~^ ERROR trait `Termination` is not implemented for `Something`
            Something
        }
    }
}

fn main() {}
