#![feature(type_alias_impl_trait)]

// check-pass

trait Tr { type Assoc; }
impl<'a> Tr for &'a str { type Assoc = &'a str; }

type OpaqueTy<'a> = impl Tr;
fn defining(s: &str) -> OpaqueTy<'_> { s }

// now we must be able to conclude `'a: 'static` from `Opaque<'a>: 'static`

fn main() {}
