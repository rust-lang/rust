// Regression test for https://github.com/rust-lang/rust/issues/160337.

fn main(... : ...)
//~^ ERROR unexpected `...`
//~| ERROR unexpected `...`
//~| ERROR expected one of `->`, `where`, or `{`, found `<eof>`
