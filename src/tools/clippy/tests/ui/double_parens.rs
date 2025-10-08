#![warn(clippy::double_parens)]
#![expect(clippy::eq_op, clippy::no_effect)]
#![feature(custom_inner_attributes)]
#![rustfmt::skip]

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

fn main() {}
