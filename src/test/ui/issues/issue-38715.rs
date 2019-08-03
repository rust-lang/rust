#[macro_export]
macro_rules! foo { ($i:ident) => {} }

#[macro_export]
macro_rules! foo { () => {} } //~ ERROR a macro named `foo` has already been exported

fn main() {}
