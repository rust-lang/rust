//@ compile-flags: -l link-arg:+bundle=arg -Z unstable-options

fn main() {}

//~? ERROR linking modifier `bundle` is only compatible with `static` linking kind
