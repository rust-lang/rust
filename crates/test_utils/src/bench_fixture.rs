//! Generates large snippets of Rust code for usage in the benchmarks.

use stdx::format_to;

pub fn big_struct() -> String {
    let n = 1_000;

    let mut buf = "pub struct RegisterBlock {".to_string();
    for i in 0..n {
        format_to!(buf, "  /// Doc comment for {}.\n", i);
        format_to!(buf, "  pub s{}: S{},\n", i, i);
    }
    buf.push_str("}\n\n");
    for i in 0..n {
        format_to!(
            buf,
            "

#[repr(transparent)]
struct S{} {{
    field: u32,
}}",
            i
        );
    }

    buf
}
