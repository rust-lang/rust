//@ aux-build:internal_unstable.rs

#![feature(allow_internal_unstable)]
#[allow(dead_code)]

#[macro_use]
extern crate internal_unstable;

struct Baz {
    #[allow_internal_unstable] //~ ERROR `allow_internal_unstable` expects a list of feature names
    baz: u8,
}

macro_rules! foo {
    ($e: expr, $f: expr) => {{
        $e;
        $f;
        internal_unstable::unstable(); //~ ERROR use of unstable
    }}
}

#[allow_internal_unstable(function)]
macro_rules! bar {
    ($e: expr) => {{
        foo!($e,
             internal_unstable::unstable());
        internal_unstable::unstable();
    }}
}

#[allow_internal_unstable(stmt_expr_attributes)]
macro_rules! internal_attr {
    ($e: expr) => {
        #[allow(overflowing_literals)]
        $e
    }
}

fn main() {
    // ok, the instability is contained.
    call_unstable_allow!();
    construct_unstable_allow!(0);
    |x: internal_unstable::Foo| { call_method_allow!(x) };
    |x: internal_unstable::Bar| { access_field_allow!(x) };
    |x: internal_unstable::Bar| { access_field_allow2!(x) }; // regression test for #77088

    // bad.
    pass_through_allow!(internal_unstable::unstable()); //~ ERROR use of unstable

    pass_through_noallow!(internal_unstable::unstable()); //~ ERROR use of unstable



    println!("{:?}", internal_unstable::unstable()); //~ ERROR use of unstable

    bar!(internal_unstable::unstable()); //~ ERROR use of unstable

    match true {
        #[allow_internal_unstable] //~ ERROR `allow_internal_unstable` expects a list of feature names
        _ => {}
    }

    assert_eq!(internal_attr!(1e100_f32), f32::INFINITY);
}
