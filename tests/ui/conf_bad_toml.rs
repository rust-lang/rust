// error-pattern: error reading Clippy's configuration file


#![plugin(clippy(conf_file="./tests/ui/conf_bad_toml.toml"))]

fn main() {}
