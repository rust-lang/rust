#[track_caller]
struct S;
//~^^ ERROR attribute cannot be used on

fn main() {}

//@ reference: attributes.codegen.track_caller.allowed-positions
