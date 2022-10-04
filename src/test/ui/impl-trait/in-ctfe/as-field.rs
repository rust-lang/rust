// For this to compile we have to reveal opaque types during ctfe without relying
// on `Reveal::All` as that would cause cycles.
//
// check-pass
#![feature(type_alias_impl_trait)]
#![crate_type = "lib"]

mod scope {
    pub type Opaque = impl Copy;

    pub const fn from_usize(x: usize) -> Opaque {
        x
    }

    pub const fn to_usize(x: Opaque) -> usize {
        x
    }
}

#[derive(Copy, Clone)]
struct Foo {
    field: scope::Opaque,
}

impl Foo {
    const fn new(field: usize) -> Self {
        Foo {
            field: scope::from_usize(field),
        }
    }

    const fn value(self) -> usize {
        scope::to_usize(self.field)
    }
}

type Cycle = impl Sized;
fn define() -> Cycle {}

fn with_opaque_in_env()
where
    Cycle: Sized,
{
    let x = [0u8; Foo::new(3).value()];
    println!("{:?}", x);
}
