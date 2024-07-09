//@ check-pass
//@ edition:2021

struct Cat;
struct Wrap<T>(T);

fn main() {
    impl From<Cat> for () {
    //~^ WARN non-local `impl` definition
        fn from(_: Cat) -> () {
            todo!()
        }
    }

    #[derive(Debug)]
    struct Elephant;

    impl From<Wrap<Wrap<Elephant>>> for () {
        fn from(_: Wrap<Wrap<Elephant>>) -> Self {
            todo!()
        }
    }
}

pub trait StillNonLocal {}

impl StillNonLocal for &str {}

fn only_global() {
    struct Foo;
    impl StillNonLocal for &Foo {}
}

struct GlobalSameFunction;

fn same_function() {
    struct Local1(GlobalSameFunction);
    impl From<Local1> for GlobalSameFunction {
        fn from(x: Local1) -> GlobalSameFunction {
            x.0
        }
    }

    struct Local2(GlobalSameFunction);
    impl From<Local2> for GlobalSameFunction {
        fn from(x: Local2) -> GlobalSameFunction {
            x.0
        }
    }
}

struct GlobalDifferentFunction;

fn diff_function_1() {
    struct Local(GlobalDifferentFunction);

    impl From<Local> for GlobalDifferentFunction {
        fn from(x: Local) -> GlobalDifferentFunction {
            x.0
        }
    }
}

fn diff_function_2() {
    struct Local(GlobalDifferentFunction);

    impl From<Local> for GlobalDifferentFunction {
        fn from(x: Local) -> GlobalDifferentFunction {
            x.0
        }
    }
}

// https://github.com/rust-lang/rust/issues/121621#issuecomment-1976826895
fn commonly_reported() {
    struct Local(u8);
    impl From<Local> for u8 {
        fn from(x: Local) -> u8 {
            x.0
        }
    }
}

// https://github.com/rust-lang/rust/issues/121621#issue-2153187542
pub trait Serde {}

impl Serde for &[u8] {}
impl Serde for &str {}

fn serde() {
    struct Thing;
    impl Serde for &Thing {}
}
