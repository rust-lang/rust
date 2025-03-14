//@ compile-flags: --crate-type dynlib
//@ error-pattern: unknown crate type: `dynlib`, expected one of: `lib`, `rlib`, `staticlib`, `dylib`, `cdylib`, `bin`, `proc-macro`

fn main() {}
