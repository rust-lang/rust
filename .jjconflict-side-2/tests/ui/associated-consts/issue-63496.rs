trait A {
    const C: usize;

    fn f() -> ([u8; A::C], [u8; A::C]);
    //~^ ERROR: E0790
    //~| ERROR: E0790
}

fn main() {}
