// Check that an expr-ctxt `include` doesn't try to parse frontmatter and instead
// treats it as a regular Rust token sequence.
//@ check-pass
#![expect(double_negations)]

fn main() {
    // issue: <https://github.com/rust-lang/rust/issues/145945>
    const _: () = assert!(-1 == include!("auxiliary/expr.rs"));
}
