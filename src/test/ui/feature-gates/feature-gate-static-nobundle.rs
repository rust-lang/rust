#[link(name = "foo", kind = "static-nobundle")]
//~^ WARNING: library kind `static-nobundle` has been superseded by specifying modifier `-bundle` with library kind `static`
//~^^ ERROR: kind="static-nobundle" is unstable
extern "C" {}

fn main() {}
