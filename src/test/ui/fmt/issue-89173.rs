// Regression test for #89173: Make sure a helpful note is issued for
// printf-style format strings using `*` to specify the width.

fn main() {
    let num = 0x0abcde;
    let width = 6;
    print!("%0*x", width, num);
    //~^ ERROR: multiple unused formatting arguments
    //~| NOTE: multiple missing formatting specifiers
    //~| NOTE: argument never used
    //~| NOTE: argument never used
    //~| NOTE: format specifiers use curly braces, and you have to use a positional or named parameter for the width
    //~| NOTE: printf formatting not supported
}
