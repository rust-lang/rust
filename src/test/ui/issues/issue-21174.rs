trait Trait<'a> {
    type A;
    type B;
}

fn foo<'a, T: Trait<'a>>(value: T::A) {
    let new: T::B = unsafe { std::mem::transmute(value) };
//~^ ERROR: transmute called with types of different sizes
}

fn main() { }
