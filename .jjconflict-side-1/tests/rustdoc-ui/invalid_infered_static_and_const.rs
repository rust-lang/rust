const FOO: dyn Fn() -> _ = ""; //~ ERROR E0121
static BOO: dyn Fn() -> _ = ""; //~ ERROR E0121
