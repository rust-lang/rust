// check-pass

// Make sure that we don't look into associated type bounds when looking for
// supertraits that define an associated type. Fixes #76593.

#![feature(associated_type_bounds)]

trait Load: Sized {
    type Blob;
}

trait Primitive: Load<Blob = Self> {}

trait BlobPtr: Primitive {}

trait CleanPtr: Load<Blob: BlobPtr> {
    fn to_blob(&self) -> Self::Blob;
}

impl Load for () {
    type Blob = Self;
}
impl Primitive for () {}
impl BlobPtr for () {}

fn main() {}
