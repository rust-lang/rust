//error-pattern: discriminator values can only be used with a c-like enum
// black and white have the same discriminator value ...

tag color {
    red = 0xff0000;
    green = 0x00ff00;
    blue = 0x0000ff;
    black = 0x000000;
    white = 0xffffff;
    other (str);
}
