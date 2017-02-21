// error-pattern: error reading Clippy's configuration file: `blacklisted-names` is expected to be a `Vec < String >` but is a `integer`

#![feature(plugin)]
#![plugin(clippy(conf_file="./tests/ui-windows/conf_bad_type.toml"))]

fn main() {}
