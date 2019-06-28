// rust-lang/rust#61402: if a struct is packed and the last
// field of the struct needs drop glue, then the compiler's well-formedness check
// will put a Sized bound on that last field.
//
// However, we do not want to ICE the compiler in our attempt to
// avoid adding that Sized bound; it is better to just let a
// potentially unneeded constraint through.

#![allow(unused_imports, dead_code)]

pub struct S;

pub trait Trait<R> { type Assoc; }

impl<X> Trait<X> for S { type Assoc = X; }

#[repr(C, packed)]
struct PackedAssocSized {
    pos: Box<<S as Trait<usize>>::Assoc>,
}

fn main() { println!("Hello, world!"); }
