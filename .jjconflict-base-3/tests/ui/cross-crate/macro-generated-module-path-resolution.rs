//@ run-pass
//@ aux-build:macro-generated-module-path-resolution.rs
//! Regression test for https://github.com/rust-lang/rust/issues/38190
//! Non inline modules parsed as macro item fragments used the wrong
//! directory for file resolution, because the fragment parser didn't
//! inherit the correct directory from the macro invocation context.

#[macro_use]
extern crate macro_generated_module_path_resolution as aux;

mod auxiliary {
    m!([
        #[path = "macro-generated-module-path-resolution"]
        mod aux;
    ]);
}

fn main() {}
