trait Trait<'a> {
    type A;
    type B;
}

fn foo<'a, T: Trait<'a>>(value: T::A) {
    let new: T::B = unsafe { std::mem::transmute(value) };
//~^ ERROR: cannot transmute between types of different sizes, or dependently-sized types
}

fn main() { }
