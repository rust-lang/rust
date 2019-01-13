enum Color {
    Red = 0xff0000,
    //~^ ERROR discriminator values can only be used with a field-less enum
    Green = 0x00ff00,
    Blue = 0x0000ff,
    Black = 0x000000,
    White = 0xffffff,
    Other(usize),
}

fn main() {}
