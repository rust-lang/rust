struct A<T>(T);

impl A<bool> {
    const B: A<u8> = Self(0);
    //~^ ERROR arguments to this function are incorrect
    //~| ERROR mismatched types
}

fn main() {}
