// rustfmt-style_edition: 2024

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
