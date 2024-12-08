//@ build-pass
#[allow(dead_code)]
static X: &'static str = &*"";
fn main() {}
