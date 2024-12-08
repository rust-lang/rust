//@ known-bug: #124352
#![rustc_never_type_options(: Unsize<U> = "hi")]

fn main() {}
