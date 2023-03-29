#![feature(type_alias_impl_trait)]

// check-pass

// Check that we cannot syntactically mention a TAIT without
// it also being part of the type.

struct DropType<T>(std::marker::PhantomData<T>);

type Foo = impl std::fmt::Debug;
// Check that, even though `Foo` is not part
// of the actual type, we do allow this to be a defining scope
#[defines(Foo)]
fn g() -> DropType<Foo> {
    let _: Foo = 42;
    DropType(std::marker::PhantomData)
}

fn main() {}
