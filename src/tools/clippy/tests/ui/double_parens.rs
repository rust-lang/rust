//@aux-build:proc_macros.rs
//@aux-build:proc_macro_derive.rs
//@aux-build:macro_rules.rs
#![warn(clippy::double_parens)]
#![expect(clippy::eq_op, clippy::no_effect)]
#![feature(custom_inner_attributes)]
#![rustfmt::skip]

use proc_macros::{external, with_span};

fn dummy_fn<T>(_: T) {}

struct DummyStruct;

impl DummyStruct {
    fn dummy_method<T>(&self, _: T) {}
}

fn simple_double_parens() -> i32 {
    ((0))
    //~^ double_parens
}

fn fn_double_parens() {
    dummy_fn((0));
    //~^ double_parens
}

fn method_double_parens(x: DummyStruct) {
    x.dummy_method((0));
    //~^ double_parens
}

fn tuple_double_parens() -> (i32, i32) {
    ((1, 2))
    //~^ double_parens
}

#[allow(clippy::unused_unit)]
fn unit_double_parens() {
    (())
    //~^ double_parens
}

fn fn_tuple_ok() {
    dummy_fn((1, 2));
}

fn method_tuple_ok(x: DummyStruct) {
    x.dummy_method((1, 2));
}

fn fn_unit_ok() {
    dummy_fn(());
}

fn method_unit_ok(x: DummyStruct) {
    x.dummy_method(());
}

// Issue #3206
fn inside_macro() {
    assert_eq!((1, 2), (1, 2), "Error");
    assert_eq!(((1, 2)), (1, 2), "Error");
    //~^ double_parens
}

fn issue9000(x: DummyStruct) {
    macro_rules! foo {
        () => {(100)}
    }
    // don't lint: the inner paren comes from the macro expansion
    (foo!());
    dummy_fn(foo!());
    x.dummy_method(foo!());

    macro_rules! baz {
        ($n:literal) => {($n)}
    }
    // don't lint: don't get confused by the expression inside the inner paren
    // having the same `ctxt` as the overall expression
    // (this is a bug that happened during the development of the fix)
    (baz!(100));
    dummy_fn(baz!(100));
    x.dummy_method(baz!(100));

    // should lint: both parens are from inside the macro
    macro_rules! bar {
        () => {((100))}
        //~^ double_parens
    }
    bar!();

    // should lint: both parens are from outside the macro;
    // make sure to suggest the macro unexpanded
    ((vec![1, 2]));
    //~^ double_parens
    dummy_fn((vec![1, 2]));
    //~^ double_parens
    x.dummy_method((vec![1, 2]));
    //~^ double_parens
}

fn issue15892() {
    use macro_rules::double_parens as double_parens_external;

    macro_rules! double_parens{
        ($a:expr, $b:expr, $c:expr, $d:expr) => {{
            let a = ($a);
            let a = (());
            //~^ double_parens
            let b = ((5));
            //~^ double_parens
            let c = std::convert::identity((5));
            //~^ double_parens
            InterruptMask((($a.union($b).union($c).union($d)).into_bits()) as u32)
        }};
    }

    // Don't lint: external macro
    (external!((5)));
    external!(((5)));

    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub struct InterruptMask(u32);

    impl InterruptMask {
        pub const OE: InterruptMask = InterruptMask(1 << 10);
        pub const BE: InterruptMask = InterruptMask(1 << 9);
        pub const PE: InterruptMask = InterruptMask(1 << 8);
        pub const FE: InterruptMask = InterruptMask(1 << 7);
        // Lint: internal macro
        pub const E: InterruptMask = double_parens!((Self::OE), Self::BE, Self::PE, Self::FE);
        // Don't lint: external macro
        pub const F: InterruptMask = double_parens_external!((Self::OE), Self::BE, Self::PE, Self::FE);
        #[allow(clippy::unnecessary_cast)]
        pub const G: InterruptMask = external!(
            InterruptMask((((Self::OE.union(Self::BE).union(Self::PE).union(Self::FE))).into_bits()) as u32)
        );
        #[allow(clippy::unnecessary_cast)]
        // Don't lint: external proc-macro
        pub const H: InterruptMask = with_span!(span
            InterruptMask((((Self::OE.union(Self::BE).union(Self::PE).union(Self::FE))).into_bits()) as u32)
        );
        pub const fn into_bits(self) -> u32 {
            self.0
        }
        #[must_use]
        pub const fn union(self, rhs: Self) -> Self {
            InterruptMask(self.0 | rhs.0)
        }
    }
}

fn issue15940() {
    use proc_macro_derive::DoubleParens;

    #[derive(DoubleParens)]
    // Don't lint: external derive macro
    pub struct Person;
}

fn main() {}
