#[rustc_doc_primitive = "usize"]
//~^ ERROR `rustc_doc_primitive` is a rustc internal attribute
/// Some docs
mod usize {}

fn main() {}
