// rustfmt-style_edition: 2024
// See https://github.com/rust-lang/rustfmt/issues/5982 and
// https://github.com/rust-lang/rustfmt/issues/5920

// https://github.com/rust-lang/rustfmt/issues/5982
struct Foo {
    #[cfg(all())]
    something_long_enough_to_wrap_foo: i32,
    #[cfg(all())]
    something_long_enough_to_wrap_bar: i32,
}

fn example() {
    let Foo {
        #[cfg(all())]
        something_long_enough_to_wrap_foo,
        #[cfg(all())]
        something_long_enough_to_wrap_bar: bar_var,
    } = Foo {
        #[cfg(all())]
        something_long_enough_to_wrap_foo: 111,
        #[cfg(all())]
        something_long_enough_to_wrap_bar: 222,
    };
}

// The same, but with leading comments on the fields.
fn example_with_comments() {
    let Foo {
        // comment on shorthand field
        #[cfg(all())]
        something_long_enough_to_wrap_foo,
        // comment on non-shorthand field
        #[cfg(all())]
        something_long_enough_to_wrap_bar: bar_var,
    } = Foo {
        #[cfg(all())]
        something_long_enough_to_wrap_foo: 111,
        #[cfg(all())]
        something_long_enough_to_wrap_bar: 222,
    };
}

// https://github.com/rust-lang/rustfmt/issues/5920
struct Demo {
    field_name_foo: (),
    field_name_bar: (),
    #[cfg(feature = "diagnostics")]
    field_name_baz: (),
    field_name_tux: (),
}

fn main() {
    let Demo {
        field_name_foo,
        #[cfg(feature = "diagnostics")]
        field_name_baz: _,
        field_name_bar,
        field_name_tux,
    } = Demo {
        field_name_foo: (),
        field_name_bar: (),
        #[cfg(feature = "diagnostics")]
        field_name_baz: (),
        field_name_tux: (),
    };
}
