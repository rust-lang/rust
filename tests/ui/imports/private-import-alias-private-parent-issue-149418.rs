//@ check-fail
//@ edition: 2024

// Private import aliases should not suggest paths through private parent modules.

mod delicious_snacks {
    use self::fruits::PEAR as fruit;

    mod fruits {
        pub const PEAR: &str = "Pear";
        pub const APPLE: &str = "Apple";
    }
}

mod delicious_snacks_without_self {
    use fruits::PEAR as fruit;

    mod fruits {
        pub const PEAR: &str = "Pear";
    }
}

fn main() {
    let _ = delicious_snacks::fruit;
    //~^ ERROR constant import `fruit` is private
    let _ = delicious_snacks_without_self::fruit;
    //~^ ERROR constant import `fruit` is private
}
