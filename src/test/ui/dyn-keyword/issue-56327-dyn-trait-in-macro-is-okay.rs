// build-pass (FIXME(62277): could be check-pass?)

// rust-lang/rust#56327: Some occurrences of `dyn` within a macro are
// not instances of identifiers, and thus should *not* be caught by the
// keyword_ident lint.
//
// Otherwise, rustfix replaces the type `Box<dyn Drop>` with
// `Box<r#dyn Drop>`, which is injecting a bug rather than fixing
// anything.

#![deny(rust_2018_compatibility)]

macro_rules! foo {
    () => {
        fn generated_foo() {
            let _x: Box<dyn Drop>;
        }
    }
}

foo!();

fn main() {
    generated_foo();
}
