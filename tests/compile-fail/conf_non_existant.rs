// error-pattern: error reading Clippy's configuration file: No such file or directory

#![feature(plugin)]
#![plugin(clippy(conf_file="./tests/compile-fail/non_existant_conf.toml"))]

fn main() {}
