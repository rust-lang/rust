// `const`s shouldn't suggest `.into()`

const TEN: u8 = 10;
const TWELVE: u16 = TEN + 2;
//~^ ERROR mismatched types [E0308]

fn main() {
    const TEN: u8 = 10;
    const ALSO_TEN: u16 = TEN;
    //~^ ERROR mismatched types [E0308]
}
