// rustfmt-trailing_comma: Never
// rustfmt-struct_lit_single_line: false
// rustfmt-struct_lit_width: 0

fn main() {
    let Foo {
        a,
        ..
    } = b;
}
