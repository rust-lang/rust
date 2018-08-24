//error-pattern:discriminant value

// black and white have the same discriminator value ...

enum color {
    red = 0xff0000,
    green = 0x00ff00,
    blue = 0x0000ff,
    black = 0x000000,
    white = 0x000000,
}

fn main() { }
