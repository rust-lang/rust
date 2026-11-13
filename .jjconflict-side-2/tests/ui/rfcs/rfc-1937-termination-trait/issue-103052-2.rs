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
