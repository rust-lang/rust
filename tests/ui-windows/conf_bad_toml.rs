// error-pattern: error reading Clippy's configuration file

#![feature(plugin)]
#![plugin(clippy(conf_file="./tests/compile-fail/conf_bad_toml.toml"))]

fn main() {}
