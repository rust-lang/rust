// Black and White have the same discriminator value ...

enum Color {
    //~^ ERROR discriminant value `0` assigned more than once
    Red = 0xff0000,
    Green = 0x00ff00,
    Blue = 0x0000ff,
    Black = 0x000000,
    White = 0x000000,
}

fn main() { }
