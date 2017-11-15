// error-pattern: error reading Clippy's configuration file


#![plugin(clippy(conf_file="../ui/conf_bad_toml.toml"))]

fn main() {}
