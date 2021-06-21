// Regression test for #75889.

const FOO: dyn Fn() -> _ = ""; //~ ERROR E0121
static BOO: dyn Fn() -> _ = ""; //~ ERROR E0121

fn main() {}
