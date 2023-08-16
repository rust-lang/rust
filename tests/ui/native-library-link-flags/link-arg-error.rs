//@compile-flags: -l link-arg:+bundle=arg -Z unstable-options
//@error-in-other-file: linking modifier `bundle` is only compatible with `static` linking kind

fn main() {}
