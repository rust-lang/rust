
enum Color {
    Red = 0xff0000,
    //~^ ERROR discriminator values can only be used with a field-less enum
    Green = 0x00ff00,
    Blue = 0x0000ff,
    Black = 0x000000,
    White = 0xffffff,
    Other (str),
    //~^ ERROR the size for values of type
    // the above is kept in order to verify that we get beyond parse errors
}

fn main() {}
