// rustfmt-trailing_comma: Never
// rustfmt-struct_lit_single_line: false

// There is an issue with how this is formatted.
// formatting should look like ./multi_line_struct_trailing_comma_never_struct_lit_width_0.rs
fn main() {
    let Foo {
        a, ..
    } = b;
}
