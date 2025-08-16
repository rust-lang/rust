fn main() {
    // INVALID: alignment (`>18`) comes after `#X`, should be `:>#18X`
    println!("{0:#X>18}", 12345);
    //~^ ERROR invalid format string: expected `>` (alignment specifier) after `:` in format string
}
