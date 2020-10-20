trait A {
    const C: usize;

    fn f() -> ([u8; A::C], [u8; A::C]);
    //~^ ERROR: type annotations needed
    //~| ERROR: type annotations needed
}

fn main() {}
