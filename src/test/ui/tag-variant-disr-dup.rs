// Black and White have the same discriminator value ...

enum Color {
    Red = 0xff0000,
    Green = 0x00ff00,
    Blue = 0x0000ff,
    Black = 0x000000,
    White = 0x000000, //~ ERROR discriminant value `0` already exists
}

fn main() { }
