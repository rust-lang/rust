#[derive("Clone")] //~ ERROR E0777
#[derive("Clone
")]
//~^^ ERROR E0777
struct Foo;

fn main() {}
