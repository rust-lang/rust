// check-pass
// compile-flags: -Z print-vtable-sizes
#![crate_type = "lib"]

trait A<T: help::V>: AsRef<[T::V]> + AsMut<[T::V]> {}

trait B<T>: AsRef<T> + AsRef<T> + AsRef<T> + AsRef<T> {}

trait C {
    fn x() {} // not object safe, shouldn't be reported
}

// This ideally should not have any upcasting cost,
// but currently does due to a bug
trait D: Send + Sync + help::MarkerWithSuper {}

// This can't have no cost without reordering,
// because `Super::f`.
trait E: help::MarkerWithSuper + Send + Sync {}

trait F {
    fn a(&self);
    fn b(&self);
    fn c(&self);

    fn d() -> Self
    where
        Self: Sized;
}

trait G: AsRef<u8> + AsRef<u16> + help::MarkerWithSuper {
    fn a(&self);
    fn b(&self);
    fn c(&self);
    fn d(&self);
    fn e(&self);

    fn f() -> Self
    where
        Self: Sized;
}

// Traits with the same name
const _: () = {
    trait S {}
};
const _: () = {
    trait S {}
};

mod help {
    pub trait V {
        type V;
    }

    pub trait MarkerWithSuper: Super {}

    pub trait Super {
        fn f(&self);
    }
}
