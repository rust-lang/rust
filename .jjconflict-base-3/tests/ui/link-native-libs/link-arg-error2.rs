//@ compile-flags: -l link-arg:+export-symbols=arg -Z unstable-options

fn main() {}

//~? ERROR linking modifier `export-symbols` is only compatible with `static` linking kind
