#[linkage = "external"]
static foo: isize = 0;
//~^^ ERROR: the `linkage` attribute is experimental and not portable

fn main() {}
