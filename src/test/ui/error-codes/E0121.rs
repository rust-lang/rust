fn foo() -> _ { 5 } //~ ERROR E0121

static BAR: _ = "test"; //~ ERROR E0121
//~^ ERROR E0121

const FOO: dyn Fn() -> _ = ""; //~ ERROR E0121

static BOO: dyn Fn() -> _ = ""; //~ ERROR E0121

fn main() {}
