#[track_caller]
fn f() {}
//~^^ ERROR the `#[track_caller]` attribute is an experimental feature

fn main() {}
