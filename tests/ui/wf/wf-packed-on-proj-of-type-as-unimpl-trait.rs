// rust-lang/rust#58158: We have special-case code to deal with case
// when a type is both packed and needs drop glue, (we move the fields
// out of their potentially unaligned locations before dropping them,
// which requires they be Sized; see PR #44884).
//
// So, we need to check if a given type needs drop-glue. That requires
// that we actually know that the concrete type, and we guard against
// the type having unknown parts (i.e. type variables) by ICE'ing in
// that scenario.
//
// But in a case where we have a projection (`Type as Trait::Assoc`)
// where `Type` does not actually implement `Trait`, we of course
// cannot have a concrete type, because there is no impl to look up
// the concrete type for the associated type `Assoc`.
//
// So, this test is just making sure that in such a case that we do
// not immediately ICE, and instead allow the underlying type error to
// surface.

pub struct Matrix<S>(S);
pub struct DefaultAllocator;

pub trait Allocator { type Buffer; }

// impl Allocator for DefaultAllocator { type Buffer = (); }

#[repr(packed)]
struct Foo(Matrix<<DefaultAllocator as Allocator>::Buffer>);
//~^ ERROR the trait bound `DefaultAllocator: Allocator` is not satisfied

fn main() { }
