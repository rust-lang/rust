// For this to compile we have to reveal opaque types during ctfe without relying
// on `Reveal::All` as that would cause cycles.
//
// check-pass
#![feature(type_alias_impl_trait)]
trait MyTrait: Copy {
    const ASSOC: usize;
}

impl MyTrait for u8 {
    const ASSOC: usize = 32;
}

const fn yeet() -> impl MyTrait {
    0u8
}

const fn output<T: MyTrait>(_: T) -> usize {
    <T as MyTrait>::ASSOC
}

type Cycle = impl Sized;
fn define() -> Cycle {}

#[repr(usize)]
enum Foo
where
    Cycle: Sized,
{
    Bar = output(yeet()),
}

fn main() {
    println!("{}", Foo::Bar as usize);
}
