#![feature(tool_lints)]

#![feature(tool_attributes)]
#![feature(stmt_expr_attributes)]

struct Outer {
    inner: u32,
}

#[deny(clippy::ref_in_deref)]
fn main() {
    let outer = Outer { inner: 0 };
    let inner = (&outer).inner;
}
