// Check error for missing writer in writeln! and write! macro
fn main() {
    let x = 1;
    let y = 2;
    write!("{}_{}", x, y);
    //~^ ERROR format argument must be a string literal
    //~| HELP you might be missing a string literal to format with
    //~| ERROR cannot write into `&'static str`
    //~| NOTE type does not implement the `write_fmt` method
    //~| HELP try adding `use std::fmt::Write;` or `use std::io::Write;` to bring the appropriate trait into scope
    writeln!("{}_{}", x, y);
    //~^ ERROR format argument must be a string literal
    //~| HELP you might be missing a string literal to format with
    //~| ERROR cannot write into `&'static str`
    //~| NOTE type does not implement the `write_fmt` method
    //~| HELP try adding `use std::fmt::Write;` or `use std::io::Write;` to bring the appropriate trait into scope
}
