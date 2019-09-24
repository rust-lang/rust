enum Color {
    Red = 0xff0000,
    //~^ ERROR custom discriminant values are not allowed in enums with tuple or struct variants
    Green = 0x00ff00,
    Blue = 0x0000ff,
    Black = 0x000000,
    White = 0xffffff,
    Other(usize),
    Other2(usize, usize),
}

fn main() {}
