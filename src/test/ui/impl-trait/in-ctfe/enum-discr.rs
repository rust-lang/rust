// This previously compiled, but was intentionally changed in #101478.
//
// See that PR for more details.
//
// check-pass
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

#[repr(usize)]
enum Foo {
    Bar = output(yeet()),
    //~^ WARNING relying on the underlying type of an opaque type in the type system
    //~| WARNING this was previously accepted by the compiler
}

fn main() {
    println!("{}", Foo::Bar as usize);
}
