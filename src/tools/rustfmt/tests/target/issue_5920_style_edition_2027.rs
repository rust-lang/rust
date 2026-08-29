// rustfmt-style_edition: 2027

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
