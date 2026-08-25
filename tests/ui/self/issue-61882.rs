struct A<T>(T);

impl A<bool> {
    const B: A<u8> = Self(0);
    //~^ ERROR mismatched types
    //~| ERROR mismatched types
}

fn main() {}
