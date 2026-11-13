// Ensure that `r#pin` doesn't get recognized as weak keyword `pin` no matter
// whether it's followed by `const` or `mut` or not.

type Ty = &r#pin const (); //~ ERROR expected one of
