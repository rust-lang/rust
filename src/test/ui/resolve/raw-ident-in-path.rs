// Regression test for issue #63882.

type A = crate::r#break; //~ ERROR cannot find type `r#break` in module `crate`

fn main() {}
