// Regression test for https://github.com/rust-lang/rust/issues/160337.

struct Baz where U : fn(() : bool)
//~^ ERROR expected identifier, found keyword `fn`
//~| ERROR expected `{` after struct name, found `<eof>`
