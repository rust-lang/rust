fn main() {}

#[allow { foo_lint } ]
//~^ ERROR wrong meta list delimiters
//~| HELP the delimiters should be `(` and `)`
fn delim_brace() {}

#[allow [ foo_lint ] ]
//~^ ERROR wrong meta list delimiters
//~| HELP the delimiters should be `(` and `)`
fn delim_bracket() {}
