fn main() {
    const PAT: u8 = 0;

    match (0, 1) {
        (PAT ..) => {}
        //~^ ERROR `X..` range patterns are not supported
        //~| ERROR exclusive range pattern syntax is experimental
        //~| ERROR mismatched types
    }
}
