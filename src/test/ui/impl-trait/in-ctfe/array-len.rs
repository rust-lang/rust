// This previously compiled, but was intentionally changed in #101478.
//
// See that PR for more details.
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

fn main() {
    let x = [0u8; output(yeet())];
    //~^ ERROR unable to use constant with a hidden value in the type system
    println!("{:?}", x);
    //~^ ERROR unable to use constant with a hidden value in the type system
}
