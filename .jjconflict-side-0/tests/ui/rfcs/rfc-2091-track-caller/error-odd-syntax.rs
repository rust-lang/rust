#[track_caller(1)]
fn f() {}
//~^^ ERROR malformed `track_caller` attribute input

fn main() {}
