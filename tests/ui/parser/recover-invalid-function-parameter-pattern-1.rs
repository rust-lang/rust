// Regression test for https://github.com/rust-lang/rust/issues/160337.

fn main(... : ...)
//~^ ERROR unexpected `...`
//~| ERROR C-variadic type `...` is not allowed in this context
//~| ERROR expected one of `->`, `where`, or `{`, found `<eof>`
