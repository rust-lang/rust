#![feature(type_alias_impl_trait)]

// Check that we cannot syntactically mention a TAIT without
// it also being part of the type.

type DropType<T> = ();
//~^ ERROR type parameter `T` is unused

type Foo = impl std::fmt::Debug;

// Check that, even though `Foo` is not part
// of the actual type, we do allow this to be a defining scope
fn g() -> DropType<Foo> {
    let _: Foo = 42;
    //~^ ERROR: constrained without being represented in the signature
    ()
}

fn main() {}
