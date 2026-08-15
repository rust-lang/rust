//@ unset-rustc-env:oopsie
//@ unset-rustc-env:a""a

env![r#"oopsie"#];
//~^ ERROR environment variable `oopsie` not defined at compile time

env![r#"a""a"#];
//~^ ERROR environment variable `a""a` not defined at compile time

fn main() {}
