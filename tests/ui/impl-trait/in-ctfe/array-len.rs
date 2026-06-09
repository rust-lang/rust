//! Check that array lengths can observe associated types of opaque types
//@ check-pass
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
    println!("{:?}", x);
}
