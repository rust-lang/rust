//@ known-bug: rust-lang/rust#126521
macro_rules! foo {
    ($val:ident) => {
        true;
    };
}

fn main() {
    #[expect(semicolon_in_expressions_from_macros)]
    let _ = foo!(x);
}
