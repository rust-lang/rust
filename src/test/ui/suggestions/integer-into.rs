fn main() {
    let a = 1u8;
    let _: i64 = a;
    //~^ ERROR mismatched types

    let b = 1i8;
    let _: isize = b;
    //~^ ERROR mismatched types

    let c = 1u8;
    let _: isize = c;
    //~^ ERROR mismatched types

    let d = 1u8;
    let _: usize = d;
    //~^ ERROR mismatched types
}
