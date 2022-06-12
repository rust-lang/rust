trait A {
    const C: usize;

    fn f() -> ([u8; A::C], [u8; A::C]);
    //~^ ERROR: E0789
    //~| ERROR: E0789
}

fn main() {}
