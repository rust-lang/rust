// unset-rustc-env:oopsie

env![r#"oopsie"#];
//~^ ERROR environment variable `oopsie` not defined at compile time

fn main() {}
