trait A {
    const C: usize;

    fn f() -> ([u8; A::C], [u8; A::C]);
    //~^ ERROR: type annotations needed: cannot resolve
    //~| ERROR: type annotations needed: cannot resolve
}

fn main() {}
