// This previously compiled, but broke with #101478.
//
// See that PR for more details.
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
