#[macro_export]
macro_rules! foo { ($i:ident) => {} }

#[macro_export]
macro_rules! foo { () => {} } //~ ERROR the name `foo` is defined multiple times

fn main() {}
