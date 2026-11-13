//! Check that compile errors are formatted in the "no-graphics" style
//! when `--error-format=human-no-graphics` is used.

//@ compile-flags: --error-format=human-no-graphics

fn foo(_: u32) {}

fn main() {
    foo("Bonjour".to_owned());
    let x = 0u32;
    x.salut();
}
