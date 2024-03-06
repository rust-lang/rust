//! Check that pattern matching can observe the hidden type of opaque types.
//@ check-pass
trait MyTrait: Copy {
    const ASSOC: u8;
}

impl MyTrait for () {
    const ASSOC: u8 = 0;
}

const fn yeet() -> impl MyTrait {}

const fn output<T: MyTrait>(_: T) -> u8 {
    <T as MyTrait>::ASSOC
}

const CT: u8 = output(yeet());

fn main() {
    match 0 {
        CT => (),
        1.. => (),
    }
}
