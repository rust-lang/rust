//@run-rustfix
#![allow(
    unused,
    clippy::needless_if,
    clippy::redundant_clone,
    clippy::derive_partial_eq_without_eq
)] // See #5700

// Define the types in each module to avoid trait impls leaking between modules.
macro_rules! impl_types {
    () => {
        #[derive(PartialEq)]
        pub struct Owned;

        pub struct Borrowed;

        impl ToOwned for Borrowed {
            type Owned = Owned;
            fn to_owned(&self) -> Owned {
                Owned {}
            }
        }

        impl std::borrow::Borrow<Borrowed> for Owned {
            fn borrow(&self) -> &Borrowed {
                static VALUE: Borrowed = Borrowed {};
                &VALUE
            }
        }
    };
}

// Only Borrowed == Owned is implemented
mod borrowed_eq_owned {
    impl_types!();

    impl PartialEq<Owned> for Borrowed {
        fn eq(&self, _: &Owned) -> bool {
            true
        }
    }

    pub fn compare() {
        let owned = Owned {};
        let borrowed = Borrowed {};

        if borrowed.to_owned() == owned {}
        if owned == borrowed.to_owned() {}
    }
}

// Only Owned == Borrowed is implemented
mod owned_eq_borrowed {
    impl_types!();

    impl PartialEq<Borrowed> for Owned {
        fn eq(&self, _: &Borrowed) -> bool {
            true
        }
    }

    fn compare() {
        let owned = Owned {};
        let borrowed = Borrowed {};

        if owned == borrowed.to_owned() {}
        if borrowed.to_owned() == owned {}
    }
}

mod issue_4874 {
    impl_types!();

    // NOTE: PartialEq<Borrowed> for T can't be implemented due to the orphan rules
    impl<T> PartialEq<T> for Borrowed
    where
        T: AsRef<str> + ?Sized,
    {
        fn eq(&self, _: &T) -> bool {
            true
        }
    }

    impl std::fmt::Display for Borrowed {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "borrowed")
        }
    }

    fn compare() {
        let borrowed = Borrowed {};

        if "Hi" == borrowed.to_string() {}
        if borrowed.to_string() == "Hi" {}
    }
}

fn main() {}
