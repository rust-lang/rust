fn main() {
    0o1.0; //~ ERROR: octal float literal is not supported
    0o2f32; //~ ERROR: octal float literal is not supported
    0o3.0f32; //~ ERROR: octal float literal is not supported
    0o4e4; //~ ERROR: octal float literal is not supported
    0o5.0e5; //~ ERROR: octal float literal is not supported
    0o6e6f32; //~ ERROR: octal float literal is not supported
    0o7.0e7f64; //~ ERROR: octal float literal is not supported
    0x8.0e+9; //~ ERROR: hexadecimal float literal is not supported
    0x9.0e-9; //~ ERROR: hexadecimal float literal is not supported
    0o; //~ ERROR: no valid digits
    1e+; //~ ERROR: expected at least one digit in exponent
    0x539.0; //~ ERROR: hexadecimal float literal is not supported
    9900000000000000000000000000999999999999999999999999999999;
    //~^ ERROR: integer literal is too large
    9900000000000000000000000000999999999999999999999999999999;
    //~^ ERROR: integer literal is too large
    0x; //~ ERROR: no valid digits
    0xu32; //~ ERROR: no valid digits
    0ou32; //~ ERROR: no valid digits
    0bu32; //~ ERROR: no valid digits
    0b; //~ ERROR: no valid digits
    0o123f64; //~ ERROR: octal float literal is not supported
    0o123.456; //~ ERROR: octal float literal is not supported
    0b101f64; //~ ERROR: binary float literal is not supported
    0b111.101; //~ ERROR: binary float literal is not supported
}
