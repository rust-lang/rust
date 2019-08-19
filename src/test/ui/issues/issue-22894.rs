// build-pass (FIXME(62277): could be check-pass?)
#[allow(dead_code)]
static X: &'static str = &*"";
fn main() {}
