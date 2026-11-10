//! Check that compile errors are formatted in the "no-graphics" style
//! when `--error-format=human-no-graphics` is used.

//@ revisions: color no-color terminal-urls
//@[color] compile-flags: --error-format=human-no-graphics --color=always
//@[no-color] compile-flags: --error-format=human-no-graphics --color=never
//@[terminal-urls] compile-flags: --error-format=human-no-graphics --color=always -Zterminal-urls=yes

fn foo(_: u32) {}

fn main() {
    foo("Bonjour".to_owned());
    let x = 0u32;
    x.salut();
}
