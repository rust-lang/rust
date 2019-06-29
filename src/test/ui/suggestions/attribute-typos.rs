#[deprcated] //~ ERROR attribute `deprcated` is currently unknown
fn foo() {}

#[tests] //~ ERROR attribute `tests` is currently unknown to the compiler
fn bar() {}

#[rustc_err] //~ ERROR attribute `rustc_err` is currently unknown
fn main() {}
