struct Rgb(u8, u8, u8);

fn main() {
    let _ = Rgb { 0, 1, 2 };
    //~^ ERROR expected identifier, found `0`
    //~| ERROR expected identifier, found `1`
    //~| ERROR expected identifier, found `2`
    //~| ERROR missing fields `0`, `1` and `2` in initializer of `Rgb`
}
