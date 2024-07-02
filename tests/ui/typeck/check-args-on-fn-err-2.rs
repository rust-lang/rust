fn main() {
    a((), 1i32 == 2u32);
    //~^ ERROR cannot find function `a`
    //~| ERROR mismatched types
}
