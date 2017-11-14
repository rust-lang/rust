// error-pattern: error reading Clippy's configuration file: `blacklisted-names` is expected to be a `Vec < String >` but is a `integer`


#![plugin(clippy(conf_file="../ui/conf_bad_type.toml"))]

fn main() {}
