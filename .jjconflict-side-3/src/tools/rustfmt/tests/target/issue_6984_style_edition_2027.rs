// rustfmt-style_edition: 2027

fn main() {
    let Demo {
        #[cfg(feature = "diagnostics")]
        // comment between attribute and field
        field_name_baz: _,

        #[cfg(feature = "diagnostics")]
        // comment between attribute and shorthand field
        field_name_bar,
    } = value;
}
