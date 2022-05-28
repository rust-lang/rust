#[link(name = "foo", kind = "static-nobundle")]
//~^ WARNING: link kind `static-nobundle` has been superseded by specifying modifier `-bundle` with link kind `static`
//~^^ ERROR: link kind `static-nobundle` is unstable
extern "C" {}

fn main() {}
