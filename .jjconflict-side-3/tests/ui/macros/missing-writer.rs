// Check error for missing writer in writeln! and write! macro
fn main() {
    let x = 1;
    let y = 2;
    write!("{}_{}", x, y);
    //~^ ERROR format argument must be a string literal
    //~| HELP you might be missing a string literal to format with
    //~| ERROR cannot write into `&'static str`
    //~| NOTE must implement `io::Write`, `fmt::Write`, or have a `write_fmt` method
    //~| HELP a writer is needed before this format string
    writeln!("{}_{}", x, y);
    //~^ ERROR format argument must be a string literal
    //~| HELP you might be missing a string literal to format with
    //~| ERROR cannot write into `&'static str`
    //~| NOTE must implement `io::Write`, `fmt::Write`, or have a `write_fmt` method
    //~| HELP a writer is needed before this format string
}
