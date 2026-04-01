//! Check that compile errors are formatted in the "short" style
//! when `--error-format=short` is used.

//@ compile-flags: --error-format=short

fn foo(_: u32) {}

fn main() {
    foo("Bonjour".to_owned());
    let x = 0u32;
    x.salut();
}
